import matplotlib.pyplot as plt
import pandas as pd

from analysis.engine.table_B1 import CAP_FAT, add_variables
from analysis.engine.table_B2 import WEIGHT, groups
from analysis.paths import FIGURES_DIR, ROOT
from analysis.utilities.cleaning import data_cutting
from analysis.utilities.survey import design_psus, mean_ci

OUTPUT = FIGURES_DIR / "figure_A1.svg"

# Short bar labels for Table B2's groups, and a colour for each block: everyone
# (navy), sex (blue), weight group (green / yellow / orange), race (light blue)
BARS = {
    "All adolescents 12–17": ("All adolescents 12–17", "#1f4e79"),
    "Males": ("Males", "#2e75b6"),
    "Females": ("Females", "#2e75b6"),
    "Healthy weight (incl. underweight)": ("Healthy weight", "#70ad47"),
    "Overweight": ("Overweight", "#ffc000"),
    "Obesity": ("Obesity", "#ed7d31"),
    "Non-Hispanic White": ("NH White", "#9dc3e6"),
    "Hispanic": ("Hispanic", "#9dc3e6"),
    "Non-Hispanic Black": ("NH Black", "#9dc3e6"),
    "Non-Hispanic Asian": ("NH Asian", "#9dc3e6"),
    "Other/Multiracial": ("Other/Multiracial", "#9dc3e6"),
}


def liver_fat_by_group(samples=None):
    # Weighted % with liver fat (95% CI) in each of Table B2's groups
    if samples is None:
        samples = data_cutting()
    psus = design_psus()
    df = add_variables(samples["screening"])
    rows = []
    for label, in_group in groups(df):
        mean, low, high = (100 * x for x in mean_ci(df, "LIVER_FAT", WEIGHT, in_group, psus))
        rows.append({"group": label, "n": in_group.sum(), "fat": mean, "low": max(low, 0), "high": high})
    return pd.DataFrame(rows).set_index("group")


def figure_a1(samples=None):
    shares = liver_fat_by_group(samples)

    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=200)
    x = range(len(shares))
    names, colours = zip(*(BARS[group] for group in shares.index))
    err = [shares["fat"] - shares["low"], shares["high"] - shares["fat"]]
    ax.bar(x, shares["fat"], color=colours, width=0.75)
    ax.errorbar(x, shares["fat"], yerr=err, fmt="none", ecolor="#333333", elinewidth=1, capsize=3)
    ax.set_xticks(list(x), names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(f"Weighted % with CAP ≥{CAP_FAT}", fontsize=10)
    ax.set_title(f"Liver fat (CAP ≥{CAP_FAT}) among U.S. adolescents 12–17, by group (95% CI)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return fig, shares


if __name__ == "__main__":
    fig, shares = figure_a1()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    print(shares.round(1).to_string())
    print(f"\nWrote {OUTPUT.relative_to(ROOT)}")
