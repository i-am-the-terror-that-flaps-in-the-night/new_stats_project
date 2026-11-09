import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.engine.table_B1 import CAP_FAT, add_variables
from analysis.engine.table_B2 import WEIGHT
from analysis.paths import ROOT
from analysis.utilities.cleaning import data_cutting
from analysis.utilities.survey import design_psus, mean_ci

OUTPUT = ROOT / "output" / "figure_A6.svg"

ALT_ADULT = 40  # U/L: the single adult cutoff many labs use
WHTR_HIGH = 0.5

DARK_BLUE, LIGHT_BLUE = "#1f4e79", "#9dc3e6"


def screens(df):
    # Each screening method (Table B3's rows): label and who it flags
    obese = df["WEIGHT_GROUP"] == "Obesity"
    high_alt = df["HIGH_ALT"] == 1
    yield "Elevated ALT, pediatric cutoffs (>26 M / >22 F)", high_alt
    yield f"Elevated ALT, common adult cutoff (>{ALT_ADULT})", df["LBXSATSI"] > ALT_ADULT
    yield "Guideline path: obesity AND elevated ALT", obese & high_alt
    yield "Obesity (BMI ≥95th percentile)", obese
    yield "Obesity OR elevated ALT", obese | high_alt
    yield "Overweight or obesity (BMI ≥85th)", df["WEIGHT_GROUP"].isin(["Overweight", "Obesity"])
    yield f"Waist-to-height ratio ≥{WHTR_HIGH}", df["WHTR"] >= WHTR_HIGH


def screen_performance(samples=None):
    # For each screen: teens flagged, sensitivity (weighted % of teens with
    # liver fat it flags) and specificity (weighted % of teens without liver fat
    # it leaves alone), each with a 95% CI
    if samples is None:
        samples = data_cutting()
    psus = design_psus()
    df = add_variables(samples["screening"])
    has_fat = df["LIVER_FAT"] == 1

    rows = []
    for label, flagged in screens(df):
        df["FLAGGED"] = flagged.astype(float)
        df["CLEARED"] = 1 - df["FLAGGED"]
        sens = [100 * x for x in mean_ci(df, "FLAGGED", WEIGHT, has_fat, psus)]
        spec = [100 * x for x in mean_ci(df, "CLEARED", WEIGHT, ~has_fat, psus)]
        rows.append({
            "screen": label, "flagged": flagged.sum(),
            "sens": sens[0], "sens_low": max(sens[1], 0), "sens_high": min(sens[2], 100),
            "spec": spec[0], "spec_low": max(spec[1], 0), "spec_high": min(spec[2], 100),
        })
    return pd.DataFrame(rows).set_index("screen")


def figure_a6(samples=None):
    performance = screen_performance(samples)
    # Best catcher at the top
    shown = performance.sort_values("sens")

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
    y = np.arange(len(shown))
    height = 0.38
    for offset, col, colour, name in [
        (height / 2, "sens", DARK_BLUE, "Sensitivity (% of teens with liver fat caught)"),
        (-height / 2, "spec", LIGHT_BLUE, "Specificity (% without liver fat correctly cleared)"),
    ]:
        err = [shown[col] - shown[f"{col}_low"], shown[f"{col}_high"] - shown[col]]
        ax.barh(y + offset, shown[col], height, color=colour, label=name)
        ax.errorbar(shown[col], y + offset, xerr=err, fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2)
        for yi, high, value in zip(y + offset, shown[f"{col}_high"], shown[col]):
            ax.text(high + 1.5, yi, f"{value:.0f}%", va="center", fontsize=7.5)

    ax.set_yticks(y, shown.index, fontsize=8)
    ax.set_xlim(0, 110)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("%", fontsize=10)
    ax.set_title(f"How well each screen identifies teens with FibroScan-measured liver fat (CAP ≥{CAP_FAT})",
                 fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return fig, performance


if __name__ == "__main__":
    fig, performance = figure_a6()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    with pd.option_context("display.width", 200):
        print(performance.round(0).to_string())
    print(f"\nWrote {OUTPUT.relative_to(ROOT)}")
