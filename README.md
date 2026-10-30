# Tensor-Based Spatiotemporal Forecasting

Traffic flow is stored as a 3-way tensor `X[day, time-of-day slot, sensor]`. CP / Tucker decomposition of the
**training** tensor gives a low-rank spatial basis and a time-of-day embedding. Temporal neural networks (GRU / LSTM / TCN)
forecast in the compact latent space (r << N sensors); predictions are lifted back to sensor space with the fixed
spatial factor.

```
X_train (D x S x N) --CP/Tucker--> A_day, A_slot (S x r_s), A_sensor (N x r_n)
window x_t (N)  --project (pinv A_sensor)-->  z_t (r_n)  --GRU/LSTM/TCN + slot embedding-->  z_hat (H x r_n)
z_hat --lift (A_sensor)--> y_hat (H x N)        loss: MSE in latent space, metrics: sensor space, original units
```

## Layout
| File | Purpose |
|---|---|
| `src/data.py` | synthetic traffic generator (daily/weekly patterns, spatial mixing, incidents), `.npz` loader for METR-LA / PEMS-BAY, windowing |
| `src/tensor_ops.py` | `TensorBasis`: CP & Tucker via TensorLy, project / lift / slot-embedding, parameter + compression accounting |
| `src/models.py` | `TensorLatentNet` (GRU/LSTM/TCN), `FullGRU`, `FullLSTM` baselines |
| `src/pipeline.py` | training w/ early stopping, MAE/RMSE @ horizons 3/6/12, memory, inference time |
| `benchmark.py` | Historical Average, Ridge-AR, full GRU/LSTM vs Tucker-GRU/LSTM/TCN and CP-GRU |
| `ablation.py` | rank sweep x {Tucker, CP}, temporal-cell ablation, trade-off plots |

## Run
```bash
pip install -r requirements.txt
python benchmark.py --epochs 30                 # synthetic data, writes results/benchmark.csv
python ablation.py --epochs 20                  # writes results/ablation.csv + ablation_tradeoff.png
# real data (array of shape (T, N) saved in an .npz):
python benchmark.py --npz metr_la.npz --key data --slots 288
```

## Design choices to mention in interviews
- **No leakage:** scaler and decomposition are fit on training days only; split is chronological by day.
- **Why latent forecasting:** the recurrent net's input/output width drops from N to r_n, cutting parameters, memory and latency.
- **Time-of-day embedding** comes from the temporal CP/Tucker factor, and is known for future steps.
- **Oracle check:** `lift(project(y_test))` error is the accuracy floor imposed by rank; the ablation shows where raising rank stops helping.
- **Fair comparison:** same windows, same early-stopping, metrics in original units, inference timed on the whole test set.

## Extensions
1. Graph-aware temporal net (replace the GRU with a diffusion-conv GRU on the latent graph) and compare.
2. Online / streaming update of factors (incremental Tucker) to handle drift.
3. Missing-data robustness: mask 20-50% of the tensor and use tensor completion before forecasting.
4. Add uncertainty (quantile loss or MC dropout) and report coverage.
5. 4-way tensor with a flow/speed/occupancy feature mode.
# tensor_based_spatiotemporal_forecasting
