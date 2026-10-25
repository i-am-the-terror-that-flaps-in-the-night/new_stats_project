import matplotlib.pyplot as plt
import pandas as pd

from analysis.paths import ROOT
from analysis.tables.cohort import add_variables
from analysis.tables.national import WEIGHT, WEIGHT_GROUP_B2
from analysis.utilities.cleaning import data_cutting

OUTPUT = ROOT / "output" / "figure_B3.png"

# Bars, top to bottom: label and whether the teen has liver fat
ROWS = [("Teens with\nliver fat", 1.0), ("Teens without\nliver fat", 0.0)]

# Short names for the bar segments (underweight is still folded into healthy
# weight, as in Table B2), and a colour for each: green, yellow, orange
SEGMENTS = {
    "Healthy weight (incl. underweight)": ("Healthy weight", "#a9d18e"),
    "Overweight": ("Overweight", "#ffd966"),
    "Obesity": ("Obesity", "#f4b183"),
}


def weight_group_shares(samples=None):
    # Weighted % of teens in each weight group, separately for teens with and
    # without liver fat. Rows: with / without; columns: weight groups.
    if samples is None:
        samples = data_cutting()
    df = add_variables(samples["screening"])
    df["GROUP"] = df["WEIGHT_GROUP"].astype(object).map(WEIGHT_GROUP_B2)

    shares = {}
    for label, fat in ROWS:
        part = df[df["LIVER_FAT"] == fat]
        totals = part.groupby("GROUP")[WEIGHT].sum()
        shares[label] = 100 * totals / totals.sum()
    return pd.DataFrame(shares).T.reindex(columns=list(SEGMENTS)).fillna(0)


def figure_b3(samples=None):
    shares = weight_group_shares(samples)

    fig, ax = plt.subplots(figsize=(9, 2.9), dpi=200)
    # Bars run top to bottom in ROWS order
    y = range(len(shares))[::-1]
    left = pd.Series(0.0, index=shares.index)
    for group, (name, colour) in SEGMENTS.items():
        widths = shares[group]
        ax.barh(y, widths, left=left, height=0.78, color=colour, edgecolor="white", linewidth=1.2)
        for row, width, start in zip(y, widths, left):
            if width >= 4:  # too narrow to hold a label
                ax.text(start + width / 2, row, f"{name}\n{width:.0f}%",
                        ha="center", va="center", fontsize=9)
        left += widths

    ax.set_yticks(list(y), shares.index, fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Weighted % of U.S. adolescents 12–17 in each group", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    # The title is the finding, worked out from the numbers
    with_fat = shares.loc[ROWS[0][0]]
    ax.set_title(
        f"{100 - with_fat['Obesity']:.0f}% of teens with liver fat do not have obesity; "
        f"{with_fat['Healthy weight (incl. underweight)']:.0f}% are a healthy weight",
        fontsize=11,
    )

    fig.tight_layout()
    return fig, shares


if __name__ == "__main__":
    fig, shares = figure_b3()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    print(shares.round(1).to_string())
    print(f"\nWrote {OUTPUT.relative_to(ROOT)}")
