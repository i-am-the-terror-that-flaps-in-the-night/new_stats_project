import pandas as pd

from analysis.engine.figures.figure_A6 import screen_performance, screens
from analysis.engine.figures.figure_B10 import weighted_auc
from analysis.engine.tables.table_B1 import add_variables
from analysis.engine.tables.table_B2 import WEIGHT, examined_teens
from analysis.paths import ROOT, TABLES_DIR
from analysis.utilities.cleaning import data_cutting
from analysis.utilities.survey import design_psus, mean_ci

OUTPUT = TABLES_DIR / "table_B3.csv"

# The measures in the "how to read this" note, ranked by AUC on the screening
# sample: column and label
AUC_MEASURES = [("LBXSATSI", "ALT"), ("BMI_Z", "BMI z-score"), ("WHTR", "waist-to-height")]


def pct(mean, low, high):
    return f"{mean:.0f}% ({max(low, 0):.0f}–{min(high, 100):.0f})"


def screening_detail(samples=None):
    # Figure A6's sensitivity and specificity, plus for each screen the chance
    # a flagged teen really has liver fat (positive predictive value) and how
    # many U.S. teens with liver fat it would miss, in millions
    if samples is None:
        samples = data_cutting()
    psus = design_psus()
    df = add_variables(samples["screening"])
    performance = screen_performance(samples)

    # U.S. teens with liver fat (millions), as in Tables B2 and B4
    population = examined_teens()[WEIGHT].sum() / 1e6
    share_fat, _, _ = mean_ci(df, "LIVER_FAT", WEIGHT, pd.Series(True, index=df.index), psus)
    with_fat = population * share_fat

    rows = []
    for label, flagged in screens(df):
        ppv = [100 * x for x in mean_ci(df, "LIVER_FAT", WEIGHT, flagged, psus)]
        p = performance.loc[label]
        # Missed = teens with liver fat times the share the screen doesn't catch;
        # a higher sensitivity means fewer missed, so the CI ends swap
        missed = [with_fat * (1 - s / 100) for s in (p["sens"], min(p["sens_high"], 100), max(p["sens_low"], 0))]
        rows.append([
            label,
            int(p["flagged"]),
            pct(p["sens"], p["sens_low"], p["sens_high"]),
            pct(p["spec"], p["spec_low"], p["spec_high"]),
            pct(*ppv),
            f"{missed[0]:.1f} ({missed[1]:.1f}–{missed[2]:.1f})",
        ])
    table = pd.DataFrame(rows, columns=[
        "Screening method",
        "Teens it flags",
        "Catches this share of teens with liver fat",
        "Correctly clears this share without it",
        "If flagged, chance of really having liver fat",
        "U.S. teens with liver fat it would miss (millions)",
    ])

    # Ranking ability (AUC) of each measure, on the screening-sample teens who
    # have all of them
    same = df.dropna(subset=[col for col, _ in AUC_MEASURES] + ["LIVER_FAT"])
    aucs = {label: weighted_auc(same[col], same["LIVER_FAT"], same[WEIGHT]) for col, label in AUC_MEASURES}
    return table, aucs


if __name__ == "__main__":
    table, aucs = screening_detail()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 50):
        print(table.to_string(index=False))
    print("\nHow to read this: Ranking ability (AUC; 0.5 is a coin flip, 1.0 is perfect): "
          + ", ".join(f"{label} {auc:.2f}" for label, auc in aucs.items()) + ".")
    print(f"\nWrote {OUTPUT.relative_to(ROOT)}")
