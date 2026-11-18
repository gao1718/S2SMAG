# S2SMAG

S2SMAG forecasts cumulative settlement during vacuum preloading. The BiGRU encoder
reads historical settlement, settlement rate, and vacuum pressure. At each forecast
step, a GRU decoder uses multi-head cross-attention to retrieve relevant historical
features and combines them with the given vacuum pressure and previous settlement.

This repository documents the 1,000-epoch experiment with C2–C8 for training,
C9 for validation, and C1 for independent testing. Each sample uses 10 historical
days and a given 10-day vacuum sequence to predict 10 days of settlement.

## Repository layout

| Path | Contents |
| --- | --- |
| [`original_data/`](original_data) | Public workbooks, variable definitions, and window preparation |
| [`model/S2SMAG/`](model/S2SMAG) | Network, configuration, training, and test commands |
| [`model/TraditionalCurves/`](model/TraditionalCurves) | C1 comparisons, native recursive forecasts, and error analysis |
| [`drawing/`](drawing) | Loss visualization from recorded training data |
| [`reference_results/`](reference_results) | Original loss and C1 prediction CSVs from the 1,000-epoch run |
| [`legacy/`](legacy) | Original repository scripts retained for reference |

## Setup

Use Python 3.11 or 3.12. Install a CUDA-enabled PyTorch build for GPU training.
The reference run used Python 3.12.10 and PyTorch 2.12.0 with CUDA 13.0.

```sh
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux/macOS or
`.\.venv\Scripts\Activate.ps1` in PowerShell. Then run this command from the
repository root.

```sh
python -m pip install -r requirements.txt
```

## Reproduce the C1/C9 experiment

The data preparation command expects a new output directory. The workbooks
contain 143 days and produce 124 overlapping windows per point. C2–C8 supply the
normalization parameters and 868 training windows. C9 and C1 each supply 124
windows. This is a spatial holdout over a shared observation period.

```sh
python -m original_data.prepare_revision --output-dir original_data/processed/c1_c9
python -m model.S2SMAG.reproduce train --data-dir original_data/processed/c1_c9 --output-dir runs/c1_c9 --epochs 1000 --device auto
python -m model.S2SMAG.reproduce test --data-dir original_data/processed/c1_c9 --checkpoint runs/c1_c9/checkpoint_best.pt --output-dir runs/c1_c9_test --device auto
```

Select `--device cuda` to require a GPU or `--device cpu` for a CPU run. The
training command writes the actual configuration, per-epoch loss history,
gradient statistics, and two local checkpoints. `checkpoint_best.pt` is selected
by minimum autoregressive C9 validation MSE. `checkpoint_final.pt` records the
last epoch. During training, teacher forcing is used with a probability of 0.8.
Validation and testing feed back the model's own predictions.

Test outputs include all C1 window–lead predictions and overall and lead-specific
metrics. The [reference CSVs](reference_results) provide the numerical outputs
of the recorded run. Trained weights are generated locally and are excluded from
version control. Hardware and PyTorch builds can yield small numerical differences.

## Analyze prediction errors and compare with traditional curves

```sh
python -m model.TraditionalCurves.reproduce --data-dir original_data/processed/c1_c9 --checkpoint runs/c1_c9/checkpoint_best.pt --output runs/c1_c9_analysis --device auto --plots
python -m drawing.plot_training_history --history reference_results/training_history.csv --output runs/c1_c9_analysis/loss_history.svg
```

The analysis uses all 1,240 C1 window–lead pairs to summarize signed errors by
forecast lead and construction phase. Phase labels follow the target date and
use the recorded vacuum thresholds documented in the output manifest. The short
and recursive forecasts are compared with hyperbolic, three-point, and Asaoka
curves fitted from observations available at each forecast origin. Recursive
forecasts use ten-day blocks, updating the settlement rate from the predicted
settlement before the next block. The analysis reports the native predictions
and their empirical errors. See the output tables and manifest for fitting
eligibility, forecast origins, and the evaluation protocol.

These experiments supply the measured forecast-period vacuum as a given
condition. A forward application requires a vacuum trajectory available at the
forecast origin, such as a preloading plan. Error quantiles describe the
evaluated windows, which overlap in time.

## Citation

Please cite *High-Precision and Generalizable Model for Long-Term Settlement
Prediction of Soft Soil Foundations Under Vacuum Preloading* and the repository
revision used for reproduction.
