"""Ablations: tensor rank x decomposition method x temporal cell, plus effect of the slot embedding.
    python ablation.py --epochs 20 --out results/ablation.csv
"""
import argparse, os, sys, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pipeline import *

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--ranks", type=int, nargs="+", default=[2, 4, 8, 12, 16])
ap.add_argument("--out", default="results/ablation.csv")
a = ap.parse_args()

set_seed(0)
D = build_data()
rows = []
for method, r, cell in itertools.product(["tucker", "cp"], a.ranks, ["gru"]):
    rank = (r, r, r) if method == "tucker" else r
    f, info = run_tensor_model(D, method, rank, cell, epochs=a.epochs)
    res = evaluate(f"{method}-{cell}", f, info, D)
    res.update(method=method, rank=r, cell=cell); rows.append(res)
    print(method, r, round(res["MAE"], 3), "compression", round(res["compression"], 1))
# temporal cell ablation at a fixed mid rank
for cell in ["gru", "lstm", "tcn"]:
    f, info = run_tensor_model(D, "tucker", (8, 8, 8), cell, epochs=a.epochs)
    res = evaluate(f"tucker-{cell}", f, info, D); res.update(method="tucker", rank=8, cell=cell, study="cell")
    rows.append(res)
df = pd.DataFrame(rows); os.makedirs(os.path.dirname(a.out), exist_ok=True); df.to_csv(a.out, index=False)

# trade-off plots
rk = df[df.study.isna()] if "study" in df else df
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for m, g in rk.groupby("method"):
    g = g.sort_values("rank")
    ax[0].plot(g["rank"], g["recon_rel_err"], "o-", label=m)
    ax[1].plot(g["rank"], g["MAE"], "o-", label=m)
    ax[2].plot(g["compression"], g["MAE"], "o-", label=m)
ax[0].set(xlabel="rank", ylabel="reconstruction rel. error", title="Information retention")
ax[1].set(xlabel="rank", ylabel="test MAE", title="Forecast accuracy vs rank")
ax[2].set(xlabel="compression ratio (x)", ylabel="test MAE", title="Accuracy vs compression"); ax[2].set_xscale("log")
for x in ax: x.legend(); x.grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(a.out), "ablation_tradeoff.png"), dpi=150)
print(df[["model", "rank", "MAE", "RMSE", "recon_rel_err", "compression", "params"]].round(3).to_string(index=False))
