"""Shared experiment code: build datasets, train neural models, evaluate with MAE/RMSE/memory/latency."""
import time, tracemalloc
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge

from data import make_synthetic_traffic, split_days, Scaler, to_matrix, make_windows
from tensor_ops import TensorBasis
from models import FullGRU, FullLSTM, TensorLatentNet

HORIZON_REPORT = (3, 6, 12)   # steps ahead reported (e.g. 15 / 30 / 60 min for 5-min data)


def set_seed(s=0):
    np.random.seed(s); torch.manual_seed(s)


def build_data(X3=None, in_len=12, horizon=12, slots_per_day=None, **gen_kw):
    """Returns dict with scaled train/val/test windows (+ the scaler and raw train tensor)."""
    if X3 is None:
        X3 = make_synthetic_traffic(**gen_kw)
    S = X3.shape[1] if slots_per_day is None else slots_per_day
    tr, va, te = split_days(X3)
    sc = Scaler(tr)                                   # statistics from TRAIN only
    out = {"scaler": sc, "train_tensor": sc.transform(tr), "slots": S, "N": X3.shape[2]}
    for name, part in (("train", tr), ("val", va), ("test", te)):
        out[name] = make_windows(to_matrix(sc.transform(part)), S, in_len, horizon)
    return out


def metrics_by_horizon(pred, true, scaler, hs=HORIZON_REPORT):
    """MAE / RMSE in ORIGINAL units at selected horizons (1-indexed steps) and averaged."""
    p, t = scaler.inverse(pred), scaler.inverse(true)
    res = {}
    for h in hs:
        e = p[:, h - 1] - t[:, h - 1]
        res[f"MAE@{h}"], res[f"RMSE@{h}"] = float(np.abs(e).mean()), float(np.sqrt((e ** 2).mean()))
    e = p - t
    res["MAE"], res["RMSE"] = float(np.abs(e).mean()), float(np.sqrt((e ** 2).mean()))
    return res


# ---------------------------------------------------------------- neural training
def _batches(arrs, bs, shuffle, rng=None):
    n = len(arrs[0]); idx = np.arange(n)
    if shuffle: (rng or np.random).shuffle(idx)
    for i in range(0, n, bs):
        j = idx[i:i + bs]
        yield [torch.as_tensor(a[j], dtype=torch.float32) for a in arrs]


def train_model(model, make_inputs, train, val, epochs=30, lr=2e-3, bs=64, patience=6, device="cpu"):
    """make_inputs(batch_arrays) -> (model_args tuple, target tensor). Early stopping on val MSE."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    loss_fn, best, bad, best_state = nn.MSELoss(), float("inf"), 0, None
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for b in _batches(train, bs, True):
            args, y = make_inputs([t.to(device) for t in b])
            loss = loss_fn(model(*args), y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        model.eval(); vl, n = 0.0, 0
        with torch.no_grad():
            for b in _batches(val, 256, False):
                args, y = make_inputs([t.to(device) for t in b])
                vl += loss_fn(model(*args), y).item() * len(y); n += len(y)
        vl /= n; sched.step(vl)
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    if best_state: model.load_state_dict(best_state)
    return {"train_time_s": time.time() - t0, "best_val_mse": best, "epochs": ep + 1}


@torch.no_grad()
def predict(model, make_inputs, arrs, to_sensor=lambda p: p, bs=256, device="cpu"):
    model.eval(); outs = []
    for b in _batches(arrs, bs, False):
        args, _ = make_inputs([t.to(device) for t in b])
        outs.append(to_sensor(model(*args)).cpu().numpy())
    return np.concatenate(outs)


def measure_inference(fn, repeats=5):
    """Mean wall-clock seconds and peak Python-heap MB (numpy) for fn(); torch tensors measured separately."""
    fn()  # warm up
    ts = []
    for _ in range(repeats):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return float(np.mean(ts))


def param_mb(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6


def peak_mem_mb(fn):
    tracemalloc.start(); fn(); _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    return peak / 1e6


# ---------------------------------------------------------------- model runners
def run_historical_average(D):
    """Predict the mean of the same time-of-day slot (from training data)."""
    tr_x, tr_y, _, tr_to = D["train"]
    S, N = D["slots"], D["N"]
    prof = np.zeros((S, N)); cnt = np.zeros(S)
    for s in range(S):
        m = tr_to == s
        prof[s] = tr_y[:, :, :][m].mean(0) if m.any() else 0
    f = lambda arrs: prof[arrs[3]]
    return f, {"params": prof.size, "param_mb": prof.nbytes / 1e6}


def run_ridge(D):
    """Linear autoregression (VAR-style): flattened window -> flattened horizon."""
    x, y, *_ = D["train"]
    m = Ridge(alpha=1.0).fit(x.reshape(len(x), -1), y.reshape(len(y), -1))
    H, N = y.shape[1], y.shape[2]
    f = lambda arrs: m.predict(arrs[0].reshape(len(arrs[0]), -1)).reshape(-1, H, N)
    return f, {"params": m.coef_.size + m.intercept_.size, "param_mb": m.coef_.nbytes / 1e6}


def run_full_neural(D, kind="gru", horizon=12, hidden=128, **tk):
    Cls = FullGRU if kind == "gru" else FullLSTM
    model = Cls(D["N"], hidden, horizon)
    mk = lambda b: ((b[0],), b[1])
    info = train_model(model, mk, D["train"][:2], D["val"][:2], **tk)
    f = lambda arrs: predict(model, mk, arrs[:2])
    info.update(params=sum(p.numel() for p in model.parameters()), param_mb=param_mb(model))
    return f, info


def run_tensor_model(D, method="tucker", rank=(8, 8, 8), cell="gru", horizon=12, hidden=64, in_len=12, **tk):
    basis = TensorBasis(method, rank).fit(D["train_tensor"])
    A = torch.as_tensor(basis.A_sensor.T, dtype=torch.float32)      # (r_n, N)
    P = torch.as_tensor(basis.P, dtype=torch.float32)               # (N, r_n)
    E = torch.as_tensor(basis.A_slot, dtype=torch.float32)          # (S, r_s)

    def mk(b):  # b = [x, y, t_in, t_out]
        x, y, ti, to = b[0], b[1], b[2].long(), b[3].long()
        return (x @ P, E[ti], E[to]), y
    model = TensorLatentNet(basis.r_n, basis.r_s, hidden, horizon, cell, in_len=in_len)
    # train in latent space against the PROJECTED target -> cheap; evaluation is in sensor space
    def mk_latent(b):
        args, y = mk(b); return args, y @ P
    info = train_model(model, mk_latent, D["train"], D["val"], **tk)
    lift = lambda z: z @ A
    f = lambda arrs: predict(model, mk, arrs, to_sensor=lift)
    info.update(params=sum(p.numel() for p in model.parameters()) + basis.n_params(),
                param_mb=param_mb(model) + basis.n_params() * 8 / 1e6,
                recon_rel_err=float(basis.rel_err_), compression=basis.full_size(D["train_tensor"]) / basis.n_params())
    return f, info


def evaluate(name, f, info, D):
    arrs = D["test"]
    pred = f(arrs)
    res = {"model": name, **metrics_by_horizon(pred, arrs[1], D["scaler"])}
    res["infer_s_per_test_set"] = measure_inference(lambda: f(arrs))
    res["infer_ms_per_sample"] = 1e3 * res["infer_s_per_test_set"] / len(arrs[0])
    res["peak_heap_mb"] = peak_mem_mb(lambda: f(arrs))
    res.update({k: v for k, v in info.items() if k in ("params", "param_mb", "train_time_s", "recon_rel_err", "compression")})
    return res
