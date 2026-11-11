import numpy as np
import pandas as pd

from analysis.engine.tables.table_B1 import format_p
from analysis.engine.tables.table_B2 import WEIGHT
from analysis.engine.tables.table_B6 import OUTCOME, RACE_DUMMIES, model_variables
from analysis.paths import ROOT, TABLES_DIR
from analysis.utilities.cleaning import data_cutting
from analysis.utilities.survey import design_psus, regression, weighted_mean_sd

OUTPUT = TABLES_DIR / "table_B14.csv"

DXA_VALID = 1  # DXXAGST: 1 = valid abdominal scan

# The three body-size measures, each put in standard deviations (weighted, among
# teens with a valid DXA scan) so they can be compared: column, new name, label
MEASURES = [
    ("BMI_Z", "BMI_SD", "BMI (per 1 SD)"),
    ("DXXVFATM", "VISCERAL_SD", "Visceral fat (per 1 SD)"),
    ("DXXSATM", "SUBCUT_SD", "Under-skin fat (per 1 SD)"),
]

# Every model also holds age, sex, race/ethnicity and the scan wand steady
ADJUST = ["RIDAGEYR", "BOY", *RACE_DUMMIES.values(), "XL"]

MODELS = [
    ("BMI only", ["BMI_SD"]),
    ("Visceral fat only", ["VISCERAL_SD"]),
    ("BMI + visceral fat", ["BMI_SD", "VISCERAL_SD"]),
    ("BMI + visceral + under-skin fat", ["BMI_SD", "VISCERAL_SD", "SUBCUT_SD"]),
]


def dxa_variables(samples=None):
    # Screening-sample teens with the model variables and the standardized fat
    # measures; teens without a valid DXA scan get missing values for those
    if samples is None:
        samples = data_cutting()
    df = model_variables(samples["screening"])
    dxa = (df["DXXAGST"] == DXA_VALID) & df["DXXVFATM"].notna() & df["DXXSATM"].notna()
    for col, new, _ in MEASURES:
        mean, sd = weighted_mean_sd(df.loc[dxa, col], df.loc[dxa, WEIGHT])
        df[new] = ((df[col] - mean) / sd).where(dxa)
    return df, dxa


def table_b14(samples=None):
    df, dxa = dxa_variables(samples)
    psus = design_psus()
    rows = []
    for name, predictors in MODELS:
        coefs, fit = regression(df, OUTCOME, [*predictors, *ADJUST], WEIGHT, psus)
        cells = []
        for _, new, _ in MEASURES:
            if new in predictors:
                est, low, high, p = coefs.loc[new]
                p = format_p(p)
                cells.append(f"{est:.0f} [{low:.0f}, {high:.0f}], p {p if p.startswith('<') else '= ' + p}")
            else:
                cells.append("—")
        rows.append([name, fit["n"], f"{fit['r2']:.2f}", *cells])
    table = pd.DataFrame(rows, columns=["Model", "Teens", "R²", *[label for *_, label in MEASURES]])

    # BMI z-score and visceral fat move together: their plain correlation
    both = df[dxa]
    correlation = np.corrcoef(both["BMI_Z"], both["DXXVFATM"])[0, 1]
    return table, correlation


if __name__ == "__main__":
    table, correlation = table_b14()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)
    with pd.option_context("display.width", 250):
        print(table.to_string(index=False))
    print(f"\nHow to read this: Visceral fat and BMI move together (correlation {correlation:.2f}), "
          f"so their numbers shrink when both are in. Each model also holds age, sex, race/ethnicity "
          f"and the scan wand steady.")
    print(f"\nWrote {OUTPUT.relative_to(ROOT)}")
