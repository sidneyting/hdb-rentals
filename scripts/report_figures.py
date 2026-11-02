"""Draw the report's EDA panel and architecture figure into report_final/figures/.

    uv run python scripts/report_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report_final" / "figures"
INK, MUTED, GRID, BLUE, PALE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#e8f1fb"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "axes.labelcolor": MUTED, "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
})


def histogram_with_box(fig, slot, values, bins, title, xlabel):
    """A histogram with a box plot on the same x axis above it."""
    inner = slot.subgridspec(2, 1, height_ratios=[1, 5], hspace=0.05)
    box, hist = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
    box.boxplot(values, orientation="horizontal", widths=0.6, patch_artist=True, showfliers=True,
                boxprops=dict(facecolor=PALE, edgecolor=BLUE, linewidth=0.8),
                medianprops=dict(color=BLUE, linewidth=1.4), whiskerprops=dict(color=BLUE, linewidth=0.8),
                capprops=dict(color=BLUE, linewidth=0.8),
                flierprops=dict(marker="o", markersize=1.5, markerfacecolor=MUTED, markeredgecolor="none", alpha=0.5))
    box.set_title(title)
    box.axis("off")
    hist.hist(values, bins=bins, color=BLUE, edgecolor="white", linewidth=0.3)
    hist.set_xlim(bins[0], bins[-1])
    box.set_xlim(bins[0], bins[-1])
    hist.set_xlabel(xlabel)
    hist.set_ylabel("Rentals")
    hist.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k" if v else "0")
    hist.grid(axis="y", color=GRID, linewidth=0.5)
    hist.set_axisbelow(True)


def grouped_boxes(ax, frame, group, title, rotate=0):
    """Rent box plots per group, ordered by median. Outliers are hidden to keep the boxes readable."""
    order = frame.groupby(group)["MONTHLY_RENT"].median().sort_values().index
    ax.boxplot([frame.loc[frame[group] == g, "MONTHLY_RENT"] for g in order], widths=0.6, patch_artist=True, showfliers=False,
               boxprops=dict(facecolor=PALE, edgecolor=BLUE, linewidth=0.7), medianprops=dict(color=BLUE, linewidth=1.3),
               whiskerprops=dict(color=BLUE, linewidth=0.7), capprops=dict(color=BLUE, linewidth=0.7))
    ax.set_xticks(range(1, len(order) + 1), order, rotation=rotate, ha="right" if rotate else "center", fontsize=6.5 if rotate else 7)
    ax.set_title(title)
    ax.set_ylabel("Monthly rent ($)")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    return order


def eda_panel():
    train = pd.read_csv(ROOT / "data" / "train.csv")
    train["FLAT_TYPE"] = train["FLAT_TYPE"].str.replace(" ", "-").str.title()
    train["TOWN"] = train["TOWN"].str.title()
    fig = plt.figure(figsize=(7.16, 5.0), layout="constrained")
    outer = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
    histogram_with_box(fig, outer[0, 0], train["MONTHLY_RENT"], list(range(200, 7801, 200)), "(a) Monthly rent", "Monthly rent ($)")
    histogram_with_box(fig, outer[0, 1], train["FLOOR_AREA_SQM"], list(range(30, 221, 5)), "(b) Floor area", "Floor area (sqm)")
    histogram_with_box(fig, outer[0, 2], train["LEASE_COMMENCE_DATE"], list(range(1965, 2026, 2)), "(c) Lease start", "Lease commencement year")
    grouped_boxes(fig.add_subplot(outer[1, 0]), train, "FLAT_TYPE", "(d) Rent by flat type", rotate=35)
    grouped_boxes(fig.add_subplot(outer[1, 1:]), train, "TOWN", "(e) Rent by town", rotate=45)
    fig.savefig(OUT / "eda_panel.png")
    plt.close(fig)
    towns = train.groupby("TOWN")["MONTHLY_RENT"].median().sort_values()
    print("town medians:", towns.iloc[0], towns.index[0], towns.iloc[-1], towns.index[-1], "n towns", len(towns))
    print("flat types:", train.groupby("FLAT_TYPE")["MONTHLY_RENT"].median().sort_values().to_dict())
    print("lease range:", train["LEASE_COMMENCE_DATE"].min(), train["LEASE_COMMENCE_DATE"].max())


def node(ax, x, y, w, h, text, accent=False, size=6.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012", mutation_aspect=1,
                                facecolor=PALE if accent else "white", edgecolor=BLUE if accent else MUTED, linewidth=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK, linespacing=1.15)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=6, color=MUTED, linewidth=0.8, shrinkA=0, shrinkB=0))


def architecture():
    fig, ax = plt.subplots(figsize=(3.5, 3.55))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0, 0.975, "(a) Pipeline", fontsize=7.5, fontweight="bold", color=INK, va="center")
    node(ax, 0, 0.71, 0.17, 0.2, "Clean\nand join\ndata")
    node(ax, 0.215, 0.71, 0.17, 0.2, "Block\nfeatures,\npremiums")
    node(ax, 0.43, 0.82, 0.2, 0.09, "GNN, MLP")
    node(ax, 0.43, 0.71, 0.2, 0.09, "LightGBM", accent=True)
    node(ax, 0.675, 0.71, 0.13, 0.2, "Blend")
    node(ax, 0.85, 0.71, 0.15, 0.2, "Holdout\nRMSE,\ngraph\ntests", size=6.2)
    node(ax, 0.43, 0.585, 0.2, 0.075, "Submission", accent=True)
    arrow(ax, (0.17, 0.81), (0.215, 0.81))
    arrow(ax, (0.385, 0.865), (0.43, 0.865))
    arrow(ax, (0.385, 0.755), (0.43, 0.755))
    arrow(ax, (0.63, 0.865), (0.675, 0.865))
    arrow(ax, (0.63, 0.755), (0.675, 0.755))
    arrow(ax, (0.805, 0.81), (0.85, 0.81))
    arrow(ax, (0.53, 0.71), (0.53, 0.66))
    ax.text(0.645, 0.6225, "refit on all training data", fontsize=6, color=MUTED, va="center")

    ax.text(0, 0.515, "(b) GNN for one rental", fontsize=7.5, fontweight="bold", color=INK, va="center")
    node(ax, 0, 0.285, 0.2, 0.18, "Block table,\n9,464 nodes,\nk = 8, 1 km", size=6)
    node(ax, 0.26, 0.285, 0.22, 0.18, "2 gated\nmessage-\npassing layers,\nsize 64", accent=True, size=6)
    node(ax, 0.54, 0.285, 0.14, 0.18, "Block\nembed-\nding", size=6.2)
    node(ax, 0.74, 0.285, 0.26, 0.18, "Head\nsize 64, ReLU,\ndropout 0.2", accent=True, size=6.2)
    node(ax, 0.46, 0.1, 0.54, 0.115, "Rental inputs: floor area, flat age,\nflat type, flat model, premiums", size=6.2)
    node(ax, 0, 0.1, 0.36, 0.115, "Rent = output\n× market level", size=6.2)
    arrow(ax, (0.2, 0.375), (0.26, 0.375))
    arrow(ax, (0.48, 0.375), (0.54, 0.375))
    arrow(ax, (0.68, 0.375), (0.74, 0.375))
    arrow(ax, (0.87, 0.215), (0.87, 0.285))
    ax.add_patch(FancyArrowPatch((0.76, 0.285), (0.76, 0.25), arrowstyle="-", color=MUTED, linewidth=0.8, shrinkA=0, shrinkB=0))
    ax.add_patch(FancyArrowPatch((0.76, 0.25), (0.18, 0.25), arrowstyle="-", color=MUTED, linewidth=0.8, shrinkA=0, shrinkB=0))
    arrow(ax, (0.18, 0.25), (0.18, 0.215))
    ax.text(0.5, 0.035, "MLP control: the same network with neighbour aggregation off.", fontsize=6.4, color=MUTED, ha="center", va="center")
    fig.savefig(OUT / "architecture.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    eda_panel()
    architecture()
