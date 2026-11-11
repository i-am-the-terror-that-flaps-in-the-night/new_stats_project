import pandas as pd

from analysis.engine.tables.table_B1 import format_p
from analysis.engine.tables.table_B6 import MODELS, OUTCOME, model_variables
from analysis.paths import ROOT, TABLES_DIR
from analysis.utilities.cleaning import data_cutting
from analysis.utilities.survey import design_psus, regression

OUTPUT = TABLES_DIR / "table_B5.csv"

# Each factor alone against liver fat, in the fasting sample (the final
# model's teens) with the fasting weight
FINAL, SAMPLE, WEIGHT, FINAL_PREDICTORS = MODELS[2]

# Rows: label and column. Each race group is its own one-variable model (that
# group = 1, every other teen = 0), which is what reproduces the doc.
ROWS = [
    ("Total sugar, per 50 g/day", "SUGAR"),
    ("Calories, per 500/day", "KCAL"),
    ("Age, per year", "RIDAGEYR"),
    ("Boy vs. girl", "BOY"),
    ("Hispanic vs. NH White", "HISPANIC"),
    ("NH Black vs. NH White", "NH_BLACK"),
    ("NH Asian vs. NH White", "NH_ASIAN"),
    ("Other vs. NH White", "OTHER_RACE"),
    ("XL wand vs. regular", "XL"),
    ("HOMA-IR, per log unit", "LOG_HOMA_IR"),
    ("BMI z-score, per SD", "BMI_Z"),
    ("Waist-to-height, per 1.0", "WHTR"),
    ("Screen time, per hour", "SCREEN_TIME"),
    ("Active days, per day", "PAQ706"),
    ("Trig/HDL, per unit", "TG_HDL"),
    ("HbA1c, per %", "LBXGH"),
    ("Uric acid, per mg/dL", "LBXSUA"),
]


def table_b5(samples=None):
    if samples is None:
        samples = data_cutting()
    psus = design_psus()
    df = model_variables(samples[SAMPLE])
    rows = []
    for label, col in ROWS:
        coefs, fit = regression(df, OUTCOME, [col], WEIGHT, psus)
        est, low, high, p = coefs.loc[col]
        rows.append([
            label, fit["n"], f"{est:+.1f} [{low:+.1f}, {high:+.1f}]", format_p(p),
            "Yes" if col in FINAL_PREDICTORS else "No",
        ])
    return pd.DataFrame(rows, columns=["Factor (alone)", "Teens", "Change in CAP [95% CI]", "p",
                                       "In the final model?"])


if __name__ == "__main__":
    table = table_b5()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)
    print(table.to_string(index=False))
    print(f"\nWrote {OUTPUT.relative_to(ROOT)}")
