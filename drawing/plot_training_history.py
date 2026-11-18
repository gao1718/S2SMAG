"""Plot the recorded loss histories on a logarithmic scale."""

import argparse
from pathlib import Path

from project_paths import configure_environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    configure_environment()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    histories = (
        ("train_mse", "Training objective (teacher forcing)", "#8796a3"),
        ("train_monitor_mse", "Training sample (autoregressive)", "#3069a5"),
        ("val_mse", "Validation C9 (autoregressive)", "#bd4255"),
    )
    data = pd.read_csv(args.history)
    required = ["epoch", *(name for name, _, _ in histories)]
    if data.empty or not set(required).issubset(data.columns):
        parser.error("The history CSV must contain epoch and all three MSE columns.")
    values = data[required].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, 1:] <= 0).any():
        parser.error("Logarithmic plotting requires finite, positive recorded losses.")
    if not (np.diff(values[:, 0]) > 0).all():
        parser.error("Epochs must be unique and increasing.")
    if args.history.resolve() == args.output.resolve():
        parser.error("Choose a figure output path separate from the input CSV.")

    figure, axis = plt.subplots(figsize=(7.2, 4.8), layout="constrained")
    for column, label, color in histories:
        axis.semilogy(data["epoch"], data[column], label=label, color=color, linewidth=1.2)
    axis.set(xlabel="Epoch", ylabel="Normalized data loss (MSE)")
    axis.grid(visible=True, which="major", color="#e2e6eb", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.legend(frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300)
    plt.close(figure)


if __name__ == "__main__":
    main()
