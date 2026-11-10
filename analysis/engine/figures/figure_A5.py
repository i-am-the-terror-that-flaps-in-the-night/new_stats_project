import matplotlib.pyplot as plt

from analysis.engine.tables.table_B1 import ALT_HIGH, CAP_FAT, add_variables
from analysis.engine.tables.table_B2 import WEIGHT
from analysis.paths import FIGURES_DIR, ROOT
from analysis.utilities.cleaning import data_cutting
from analysis.utilities.survey import design_psus, mean_ci

OUTPUT = FIGURES_DIR / "figure_A5.svg"

# Dot colour by weight group (underweight folded into healthy weight)
GROUPS = {"Healthy weight": "#70ad47", "Overweight": "#ffc000", "Obesity": "#ed7d31"}
RED, BLUE = "#c0392b", "#2e75b6"


def alt_vs_liver_fat(samples=None):
    # Every screening-sample teen's ALT and CAP, plus the numbers the figure
    # quotes: the weighted share of teens with liver fat whose ALT is normal,
    # and how many teens have a high ALT and how many of those have no liver fat
    if samples is None:
        samples = data_cutting()
    df = add_variables(samples["screening"])
    df["GROUP"] = df["WEIGHT_GROUP"].astype(object).replace({"Underweight": "Healthy weight"})
    df["NORMAL_ALT"] = 1 - df["HIGH_ALT"]
    normal, _, _ = mean_ci(df, "NORMAL_ALT", WEIGHT, df["LIVER_FAT"] == 1, design_psus())
    high = df["HIGH_ALT"] == 1
    counts = {
        "normal_alt_share": 100 * normal,
        "high_alt": high.sum(),
        "high_alt_no_fat": (high & (df["LIVER_FAT"] == 0)).sum(),
    }
    return df, counts


def figure_a5(samples=None):
    df, counts = alt_vs_liver_fat(samples)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    for group, colour in GROUPS.items():
        part = df[df["GROUP"] == group]
        ax.scatter(part["LBXSATSI"], part["LUXCAPM"], s=9, color=colour, alpha=0.65,
                   linewidths=0, label=f"{group} (n={len(part)})")
    ax.set_xscale("log")

    ax.axhline(CAP_FAT, color=RED, linestyle="--", linewidth=1)
    ax.text(0.99, CAP_FAT + 4, f"Liver fat cutoff (CAP {CAP_FAT})", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8, color=RED)
    # Teen ALT cutoffs differ by sex: girls dotted, boys dashed
    girls, boys = ALT_HIGH[2], ALT_HIGH[1]
    ax.axvline(girls, color=BLUE, linestyle=":", linewidth=1)
    ax.axvline(boys, color=BLUE, linestyle="--", linewidth=1)
    ax.text(boys * 1.05, 0.97, f"ALT cutoffs\n({girls} girls, {boys} boys)", transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=8, color=BLUE)
    ax.text(0.03, 0.97, f"Upper-left: liver fat with normal ALT\n{counts['normal_alt_share']:.0f}% of teens "
            f"with liver fat are here", transform=ax.transAxes, ha="left", va="top", fontsize=8, color="#333333")

    ax.set_xlabel("ALT, U/L (log scale)", fontsize=10)
    ax.set_ylabel("Liver fat, CAP (dB/m)", fontsize=10)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return fig, counts


if __name__ == "__main__":
    fig, counts = figure_a5()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    print(f"Teens with liver fat whose ALT is normal: {counts['normal_alt_share']:.0f}% (weighted)")
    print(f"Teens with a high ALT: {counts['high_alt']}, of whom {counts['high_alt_no_fat']} "
          f"({100 * counts['high_alt_no_fat'] / counts['high_alt']:.0f}%) have no liver fat")
    print(f"\nWrote {OUTPUT.relative_to(ROOT)}")
