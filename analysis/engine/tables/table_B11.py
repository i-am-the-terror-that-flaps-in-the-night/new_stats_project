import pandas as pd

from analysis.engine.tables.table_B1 import format_p
from analysis.engine.tables.table_B6 import MODELS, fit_models
from analysis.paths import ROOT, TABLES_DIR
from analysis.utilities.cleaning import data_cutting

OUTPUT = TABLES_DIR / "table_B11.csv"

MODEL_A, _, MODEL_B_BMI = (name for name, *_ in MODELS)

# Each row: the check, which diet days, and the model (the same fits as Figure B5)
ROWS = [
    ("One day (main)", False, MODEL_A, "Model A"),
    ("One day (main) ★", False, MODEL_B_BMI, "Model B with BMI"),
    ("Two-day average", True, MODEL_A, "Model A"),
    ("Two-day average", True, MODEL_B_BMI, "Model B with BMI"),
]


def table_b11(samples=None):
    # The sugar effect with one day of diet against the average of two days
    if samples is None:
        samples = data_cutting()
    fits = {two_day: fit_models(samples, two_day) for two_day in (False, True)}
    rows = []
    for check, two_day, model, short in ROWS:
        coefs, fit = fits[two_day][model]
        est, low, high, p = coefs.loc["SUGAR"]
        rows.append([check, short, fit["n"], f"{est:+.1f} [{low:+.1f}, {high:+.1f}]", format_p(p)])
    return pd.DataFrame(rows, columns=["Check", "Model", "Teens", "Sugar effect per 50 g/day [95% CI]", "p"])


if __name__ == "__main__":
    table = table_b11()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)
    print(table.to_string(index=False))
    print(f"\nWrote {OUTPUT.relative_to(ROOT)}")
