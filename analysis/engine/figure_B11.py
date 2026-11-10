import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.engine.table_B1 import CAP_FAT, add_variables, format_p
from analysis.engine.table_B2 import WEIGHT
from analysis.paths import FIGURES_DIR, ROOT
from analysis.utilities.cleaning import data_cutting
from analysis.utilities.survey import design_psus, mean_ci, regression

OUTPUT = FIGURES_DIR / "figure_B11.svg"

VISCERAL = "DXXVFATM"  # visceral (belly) fat mass from the DXA scan, grams
DXA_VALID = 1  # DXXAGST: 1 = valid abdominal scan

# Weight groups (underweight folded into healthy weight, as in Table B2) and
# their colours, the same green / yellow / orange as Figure B3
GROUPS = {
    "Healthy weight": "#a9d18e",
    "Overweight": "#ffd966",
    "Obesity": "#f4b183",
}
THIRDS = ["Lowest third", "Middle third", "Highest third"]


def weighted_cuts(x, w, shares):
    # Values where the weighted share of U.S. teens reaches each of `shares`
    order = np.argsort(x.to_numpy())
    values = x.to_numpy()[order]
    share = np.cumsum(w.to_numpy()[order]) / w.sum()
    return [values[np.searchsorted(share, s)] for s in shares]


def visceral_thirds(samples=None):
    # Screening-sample teens with a valid DXA scan, split into visceral-fat
    # thirds within their own weight group (survey-weighted, so each third
    # stands for a third of U.S. teens in that group). One row per group and
    # third: teens, average visceral fat, CAP and % with liver fat (95% CI).
    # Also the trend p-value in each group: does CAP climb across the thirds?
    if samples is None:
        samples = data_cutting()
    psus = design_psus()
    df = add_variables(samples["screening"])
    df["GROUP"] = df["WEIGHT_GROUP"].astype(object).replace({"Underweight": "Healthy weight"})
    dxa = (df["DXXAGST"] == DXA_VALID) & df[VISCERAL].notna()

    df["THIRD"] = np.nan
    for group in GROUPS:
        in_group = dxa & (df["GROUP"] == group)
        cuts = weighted_cuts(df.loc[in_group, VISCERAL], df.loc[in_group, WEIGHT], [1 / 3, 2 / 3])
        df.loc[in_group, "THIRD"] = pd.cut(df.loc[in_group, VISCERAL], [-np.inf, *cuts, np.inf],
                                           labels=[1, 2, 3]).astype(float)

    rows, trend = [], {}
    for group in GROUPS:
        for third, label in enumerate(THIRDS, start=1):
            at = (df["GROUP"] == group) & (df["THIRD"] == third)
            cap = mean_ci(df, "LUXCAPM", WEIGHT, at, psus)
            fat = mean_ci(df, "LIVER_FAT", WEIGHT, at, psus)
            rows.append({
                "group": group, "third": label, "n": at.sum(),
                "visceral": np.average(df.loc[at, VISCERAL], weights=df.loc[at, WEIGHT]),
                "cap": cap[0], "cap_low": cap[1], "cap_high": cap[2],
                # A share can't be below 0% or above 100%
                "fat": 100 * fat[0], "fat_low": 100 * max(fat[1], 0), "fat_high": 100 * min(fat[2], 1),
            })
        # Trend within the group: weighted regression of CAP on the third (1-3).
        # CAP outside the group is set to missing, so only the group is fitted
        # but every teen's PSU still counts (domain estimation).
        in_group = df.assign(Y=df["LUXCAPM"].where(df["GROUP"] == group))
        coefs, _ = regression(in_group, "Y", ["THIRD"], WEIGHT, psus)
        trend[group] = coefs.loc["THIRD", "p"]
    return pd.DataFrame(rows).set_index(["group", "third"]), trend


def figure_b11(samples=None):
    thirds, trend = visceral_thirds(samples)

    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=200)
    width = 0.26
    for i, (group, colour) in enumerate(GROUPS.items()):
        part = thirds.loc[group]
        x = np.arange(len(THIRDS)) + (i - 1) * width
        err = [part["fat"] - part["fat_low"], part["fat_high"] - part["fat"]]
        p = format_p(trend[group])
        ax.bar(x, part["fat"], width, color=colour, edgecolor="#555555",
               label=f"{group} (CAP trend p {p if p.startswith('<') else '= ' + p})")
        ax.errorbar(x, part["fat"], yerr=err, fmt="none", ecolor="#333333", elinewidth=0.9, capsize=2.5)
        for xi, n in zip(x, part["n"]):
            ax.text(xi, 1.5, f"n={n}", ha="center", va="bottom", fontsize=7, color="#333333")

    ax.set_xticks(range(len(THIRDS)), THIRDS, fontsize=10)
    ax.set_xlabel("Visceral (belly) fat, within each weight group", fontsize=10)
    ax.set_ylabel(f"Weighted % with liver fat (CAP ≥{CAP_FAT})", fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_title("Same weight group, different belly fat: liver fat by visceral-fat third", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return fig, thirds, trend


if __name__ == "__main__":
    fig, thirds, trend = figure_b11()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    with pd.option_context("display.width", 200):
        print(thirds.round(1).to_string())
    for group, p in trend.items():
        print(f"CAP trend across thirds, {group}: p {format_p(p)}")
    print(f"\nWrote {OUTPUT.relative_to(ROOT)}")
