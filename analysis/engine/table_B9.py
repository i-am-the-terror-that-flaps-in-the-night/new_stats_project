import pandas as pd

from analysis.engine.table_B1 import CAP_FAT
from analysis.engine.table_B6 import MODELS, OUTCOME, SUGAR_UNIT, model_variables
from analysis.paths import ROOT, TABLES_DIR
from analysis.utilities.cleaning import data_cutting
from analysis.utilities.survey import design_psus, regression, weighted_mean_sd

OUTPUT = TABLES_DIR / "table_B9.csv"

# The primary test: sugar in Model B with BMI
PRIMARY, SAMPLE, WEIGHT, PREDICTORS = MODELS[2]

# Extra sugar every day (g/day) and how to say it
AMOUNTS = [
    (39, "39 g/day (one can of regular soda)"),
    (50, "50 g/day"),
    (100, "100 g/day (about 2.5 cans)"),
]


def soda_scale(samples=None):
    # The sugar effect from the primary model, per gram, with the numbers that
    # put it in context: CAP's spread and average in the same teens, and what
    # one SD of BMI z-score is worth in the same model
    if samples is None:
        samples = data_cutting()
    df = model_variables(samples[SAMPLE])
    coefs, _ = regression(df, OUTCOME, PREDICTORS, WEIGHT, design_psus())
    in_fit = df[[OUTCOME, *PREDICTORS]].notna().all(axis=1)
    mean, sd = weighted_mean_sd(df.loc[in_fit, OUTCOME], df.loc[in_fit, WEIGHT])
    per_gram = coefs.loc["SUGAR", ["estimate", "low", "high"]] / SUGAR_UNIT
    return per_gram, sd, mean, coefs.loc["BMI_Z", "estimate"]


def table_b9(samples=None):
    per_gram, sd, _, _ = soda_scale(samples)
    rows = []
    for grams, label in AMOUNTS:
        est, low, high = grams * per_gram
        rows.append([label, f"{est:+.1f} ({low:+.1f} to {high:+.1f})", f"{100 * est / sd:+.0f}% of one SD"])
    return pd.DataFrame(rows, columns=[
        "Extra sugar every day",
        "Change in liver fat, dB/m (95% CI)",
        "Compared with the spread of CAP",
    ])


if __name__ == "__main__":
    samples = data_cutting()
    table = table_b9(samples)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)
    with pd.option_context("display.width", 250):
        print(table.to_string(index=False))
    _, sd, mean, bmi = soda_scale(samples)
    print(f"\nHow to read this: the spread (SD) of CAP is about {sd:.0f} dB/m, and the average teen "
          f"is {CAP_FAT - mean:.0f} dB/m below the cutoff. For comparison, one SD of BMI z-score "
          f"is worth about {bmi:.0f} dB/m.")
    print(f"\nWrote {OUTPUT.relative_to(ROOT)}")
