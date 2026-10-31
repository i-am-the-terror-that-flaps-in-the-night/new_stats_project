import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.engine.table_B1 import CAP_FAT, format_p
from analysis.engine.table_B6 import MODELS, model_variables
from analysis.paths import ROOT
from analysis.utilities.cleaning import data_cutting
from analysis.utilities.survey import design_psus, mean_ci, regression

OUTPUT = ROOT / "output" / "figure_B7.svg"

WEIGHT = "WTDRD1"  # day 1 diet weight: the quarters come from the diet sample
SUGAR = "DR1TSUGR"

# The trend test adjusted for everything in Model A except sugar itself
_, _, _, MODEL_A = MODELS[0]
ADJUST = [col for col in MODEL_A if col != "SUGAR"]

# Light fill and dark edge for each panel: blue for CAP, orange for liver fat
CAP_COLOURS = ("#9dc3e6", "#1f4e79")
FAT_COLOURS = ("#f4b183", "#843c0c")


def weighted_quartile_cuts(x, w):
    # Sugar values where the weighted share of U.S. teens reaches 25%, 50% and
    # 75%, so each quarter stands for about a quarter of U.S. teens (not of the
    # teens in the sample, which is why the quarters hold different numbers)
    order = np.argsort(x.to_numpy())
    values = x.to_numpy()[order]
    share = np.cumsum(w.to_numpy()[order]) / w.sum()
    return [values[np.searchsorted(share, q)] for q in (0.25, 0.5, 0.75)]


def sugar_quarters(samples=None):
    # One row per sugar quarter (sugar range, teens, average sugar, CAP and %
    # with liver fat, each with a 95% CI), plus the trend-test p-values
    if samples is None:
        samples = data_cutting()
    psus = design_psus()
    df = model_variables(samples["diet"])
    cuts = weighted_quartile_cuts(df[SUGAR], df[WEIGHT])
    df["QUARTER"] = pd.cut(df[SUGAR], [-np.inf, *cuts, np.inf], labels=[1, 2, 3, 4]).astype(float)

    rows = []
    for quarter in (1, 2, 3, 4):
        in_q = df["QUARTER"] == quarter
        sugar = df.loc[in_q, SUGAR]
        cap = mean_ci(df, "LUXCAPM", WEIGHT, in_q, psus)
        fat = mean_ci(df, "LIVER_FAT", WEIGHT, in_q, psus)
        rows.append({
            "quarter": f"Q{quarter}",
            "range": f"{sugar.min():.0f}–{sugar.max():.0f}",
            "n": in_q.sum(),
            "sugar": np.average(sugar, weights=df.loc[in_q, WEIGHT]),
            "cap": cap[0], "cap_low": cap[1], "cap_high": cap[2],
            "fat": 100 * fat[0], "fat_low": 100 * max(fat[1], 0), "fat_high": 100 * fat[2],
        })

    # Trend test: does the outcome climb with the quarter number (1-4)? A
    # weighted regression on the quarter number, alone and with Model A's factors
    trend = {}
    for outcome in ("LUXCAPM", "LIVER_FAT"):
        alone, _ = regression(df, outcome, ["QUARTER"], WEIGHT, psus)
        adjusted, _ = regression(df, outcome, ["QUARTER", *ADJUST], WEIGHT, psus)
        trend[outcome] = (alone.loc["QUARTER", "p"], adjusted.loc["QUARTER", "p"])
    return pd.DataFrame(rows).set_index("quarter"), trend


def trend_title(name, p):
    alone, adjusted = (format_p(x) for x in p)
    return f"{name}\ntrend p = {alone} (adjusted p = {adjusted})"


def bars(ax, quarters, col, colours):
    fill, edge = colours
    x = range(len(quarters))
    err = [quarters[col] - quarters[f"{col}_low"], quarters[f"{col}_high"] - quarters[col]]
    ax.bar(x, quarters[col], width=0.8, color=fill, edgecolor=edge)
    ax.errorbar(x, quarters[col], yerr=err, fmt="none", ecolor=edge, elinewidth=1.4, capsize=4)
    ticks = [f"{q}\n{r} g" for q, r in zip(quarters.index, quarters["range"])]
    ax.set_xticks(list(x), ticks, fontsize=8)
    ax.set_xlabel("Total sugar quartile (g/day)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)


def figure_b7(samples=None):
    quarters, trend = sugar_quarters(samples)

    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.8), dpi=200)
    bars(left, quarters, "cap", CAP_COLOURS)
    # CAP bars start at 150, not 0: no teen's CAP is anywhere near 0
    left.set_ylim(150, 260)
    left.set_ylabel("Weighted mean CAP (dB/m)", fontsize=10)
    left.set_title(trend_title("Liver fat (CAP)", trend["LUXCAPM"]), fontsize=10)

    bars(right, quarters, "fat", FAT_COLOURS)
    right.set_ylim(0, None)
    right.set_ylabel(f"Weighted % with CAP ≥{CAP_FAT}", fontsize=10)
    right.set_title(trend_title("Liver fat above cutoff", trend["LIVER_FAT"]), fontsize=10)

    fig.tight_layout()
    return fig, quarters, trend


if __name__ == "__main__":
    fig, quarters, trend = figure_b7()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    with pd.option_context("display.width", 200):
        print(quarters.round(1).to_string())
    for outcome, (alone, adjusted) in trend.items():
        print(f"Trend {outcome}: p = {format_p(alone)}, adjusted p = {format_p(adjusted)}")
    print(f"\nWrote {OUTPUT.relative_to(ROOT)}")
