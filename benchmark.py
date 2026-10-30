"""Benchmark tensor-based models against conventional forecasters.
    python benchmark.py --epochs 30 --out results/benchmark.csv [--npz path --key data --slots 288]
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import pandas as pd
from data import load_npz
from pipeline import *

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--npz"); ap.add_argument("--key", default="data"); ap.add_argument("--slots", type=int, default=288)
ap.add_argument("--out", default="results/benchmark.csv")
a = ap.parse_args()

set_seed(0)
X3 = load_npz(a.npz, a.key, a.slots) if a.npz else None
D = build_data(X3, in_len=12, horizon=12)
tk = dict(epochs=a.epochs)
r = a.rank
runs = [
    ("HistoricalAverage", *run_historical_average(D)),
    ("Ridge-AR", *run_ridge(D)),
    ("FullGRU", *run_full_neural(D, "gru", **tk)),
    ("FullLSTM", *run_full_neural(D, "lstm", **tk)),
    (f"Tucker-GRU(r={r})", *run_tensor_model(D, "tucker", (r, r, r), "gru", **tk)),
    (f"Tucker-LSTM(r={r})", *run_tensor_model(D, "tucker", (r, r, r), "lstm", **tk)),
    (f"Tucker-TCN(r={r})", *run_tensor_model(D, "tucker", (r, r, r), "tcn", **tk)),
    (f"CP-GRU(R={r})", *run_tensor_model(D, "cp", r, "gru", **tk)),
]
rows = []
for name, f, info in runs:
    if isinstance(info, dict) and "params" not in info: info = {}
    rows.append(evaluate(name, f, info, D)); print(rows[-1]["model"], round(rows[-1]["MAE"], 3), round(rows[-1]["RMSE"], 3))
os.makedirs(os.path.dirname(a.out), exist_ok=True)
df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
print(df[["model", "MAE", "RMSE", "MAE@3", "MAE@6", "MAE@12", "params", "peak_heap_mb", "infer_ms_per_sample"]].round(3).to_string(index=False))
