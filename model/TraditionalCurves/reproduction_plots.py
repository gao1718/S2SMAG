"""Render C1 comparison and error figures directly from the analysis tables."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


OBSERVED = "#BF4355"
VACUUM = "#55575C"
METHODS = {
    "S2SMAG": ("S2SMAG", "#2C67A2", "-"),
    "Hyperbolic": ("Hyperbolic", "#B88026", "--"),
    "ThreePoint": ("Three-point", "#815A9B", "-."),
    "Asaoka": ("Asaoka", "#358984", ":"),
}


def export(figure, destination, name):
    figure.savefig(destination / f"{name}.svg")
    figure.savefig(destination / f"{name}.png", dpi=800)
    plt.close(figure)


def comparison_figure(destination, observations, comparisons, task):
    groups = comparisons.loc[(comparisons.task == task) & (comparisons.protocol == "first80")]
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.8))
    figure.subplots_adjust(left=.08, right=.92, bottom=.19, top=.78, wspace=.27)
    legend = [Line2D([0], [0], color=OBSERVED, label="Observed")]
    legend += [Line2D([0], [0], color=color, linestyle=style, label=label)
               for label, color, style in METHODS.values()]
    legend.append(Line2D([0], [0], color=VACUUM, linestyle=":", label="Measured vacuum"))
    figure.legend(handles=legend, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, .99))
    for index, (axis, (case, values)) in enumerate(zip(axes, groups.groupby("case", sort=False), strict=True)):
        cutoff = int(values.history_days.iloc[0])
        start = cutoff - 10 if task == "short" else 0
        stop = int(values.time_index.max()) + 1
        observed = observations.iloc[start:stop]
        boundary = observations.date.iloc[cutoff - 1] + (
            observations.date.iloc[cutoff] - observations.date.iloc[cutoff - 1]
        ) / 2
        axis.axvspan(observed.date.iloc[0], boundary, facecolor="#E9F1F7", linewidth=0)
        axis.axvline(boundary, color="#58616A", linestyle="--", linewidth=.8)
        axis.plot(observed.date, observed.settlement, color=OBSERVED, linewidth=1.5, zorder=3)
        vacuum_axis = axis.twinx()
        vacuum_axis.plot(observed.date, observed.vacuum, linestyle=":", color=VACUUM, linewidth=.95)
        vacuum_axis.set(ylim=(0, 100), yticks=[0, 50, 100])
        vacuum_axis.tick_params(labelright=index == 2)
        if index == 2:
            vacuum_axis.set_ylabel("Vacuum pressure (kPa)")
        for method, group in values.loc[values.ranking_eligible].groupby("method", sort=False):
            _, color, style = METHODS[method]
            axis.plot(group.date, group.predicted, color=color, linestyle=style, linewidth=1.35, zorder=5)
        axis.set(xlim=(observed.date.iloc[0], observed.date.iloc[-1]),
                 title=f"({chr(97 + index)}) {case}", xlabel="Date")
        tick_indices = np.linspace(0, len(observed) - 1, 4, dtype=int)
        axis.set_xticks(observed.date.iloc[tick_indices])
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        axis.grid(color="#DEE5EC", linewidth=.6)
        axis.set_axisbelow(True)
        if index == 0:
            axis.set_ylabel("Cumulative settlement (mm)")
    export(figure, destination, f"classical_comparison_{task}")


def error_figure(destination, windows):
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), sharey=True, layout="constrained")
    rng = np.random.default_rng(42)
    configurations = [
        ("step", "(a) Errors by forecast lead", ["#2C67A2"] * 10),
        ("phase", "(b) Errors by construction phase", ["#B88026", "#2C67A2", "#BF4355"]),
    ]
    for axis, (group_key, title, colors) in zip(axes, configurations, strict=True):
        groups = list(windows.groupby(group_key, sort=False))
        values = [group.error.to_numpy() for _, group in groups]
        labels = [str(key).replace("Maintained vacuum", "Maintained\nvacuum")
                  .replace("Vacuum loss response", "Vacuum loss") for key, _ in groups]
        boxes = axis.boxplot(values, widths=.42, showfliers=False, patch_artist=True,
                             medianprops={"color": "#222222", "linewidth": 1.1})
        for position, (box, data, color) in enumerate(zip(boxes["boxes"], values, colors, strict=True), 1):
            box.set(facecolor=matplotlib.colors.to_rgba(color, .16), edgecolor=color)
            axis.scatter(position + rng.uniform(-.15, .15, len(data)), data, s=4, color=color,
                         alpha=.23, linewidths=0)
        axis.axhline(0, color="#5A656F", linewidth=.85)
        axis.set_xticks(range(1, len(labels) + 1), labels)
        axis.set_title(title, loc="left")
        axis.grid(axis="y", color="#DEE5EC", linewidth=.6)
        axis.set_axisbelow(True)
    axes[0].set(xlabel="Forecast lead (day)", ylabel="Predicted − observed (mm)")
    export(figure, destination, "forecast_errors")


def save_plots(destination, observations, comparisons, windows):
    with matplotlib.rc_context({"font.size": 9, "axes.titlesize": 10, "svg.fonttype": "none"}):
        for task in ("short", "long"):
            comparison_figure(destination, observations, comparisons, task)
        error_figure(destination, windows)
