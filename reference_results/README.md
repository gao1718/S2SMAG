# Recorded C1/C9 reference run

These three CSV files are the original numerical outputs of the 1,000-epoch run
with C2–C8 for training, C9 for validation, and C1 for testing. The split identifier is
`eb78c87240c48e864d9a3eaa2040e569a1a1037bbcc901af2730c39d72d9bb27`.
The loss history contains the recorded values for all 1,000 epochs.

- `train_mse` measures the training objective with teacher forcing.
- `train_monitor_mse` measures autoregressive predictions for a fixed subset of 35 training windows.
- `val_mse` measures autoregressive predictions for all 124 C9 validation windows.
- The corresponding `_mm2` columns are expressed in mm².

The prediction tables contain 124 windows and 10 lead times per window. Their
`actual`, `predicted`, and `error` columns are in mm.
The `actual` values are the inverse-transformed numerical targets used in the original test.

| File | Checkpoint rule | Epoch | C1 MAE (mm) | C1 RMSE (mm) |
| --- | --- | ---: | ---: | ---: |
| `c1_best_validation_predictions.csv` | Minimum autoregressive C9 validation MSE | 967 | 2.8982 | 4.2499 |
| `c1_final_epoch_predictions.csv` | Fixed training endpoint | 1,000 | 1.4631 | 2.6861 |

Use C9 to select a checkpoint. The fixed endpoint is included to document the
training run. Both files use the measured C1 vacuum during the forecast window as a
given input.
