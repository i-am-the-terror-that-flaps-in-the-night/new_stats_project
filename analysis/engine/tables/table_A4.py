import pandas as pd

from analysis.engine.tables.table_B1 import CAP_FAT, add_variables
from analysis.paths import ROOT, TABLES_DIR
from analysis.utilities.cleaning import data_cutting
from analysis.utilities.survey import weighted_mean_sd

OUTPUT = TABLES_DIR / "table_A4.csv"

# The poster's mini table: a few weighted numbers describing the teens. The
# full version (boys and girls apart, fasting tests, DXA) is Table B1.
# Each row: label, sample, weight, column, decimals, and "mean" or "pct"
# (pct = weighted % of a yes/no column).
ROWS = [
    ("Age, years", "screening", "WTMEC2YR", "RIDAGEYR", 1, "mean"),
    ("Boys", "screening", "WTMEC2YR", "BOY", 0, "pct"),
    ("Hispanic", "screening", "WTMEC2YR", "HISPANIC", 0, "pct"),
    ("Obesity (BMI ≥95th percentile)", "screening", "WTMEC2YR", "OBESE", 0, "pct"),
    ("Liver fat (CAP), average", "screening", "WTMEC2YR", "LUXCAPM", 0, "mean"),
    (f"Liver fat present (CAP ≥{CAP_FAT})", "screening", "WTMEC2YR", "LIVER_FAT", 0, "pct"),
    ("ALT, average", "screening", "WTMEC2YR", "LBXSATSI", 1, "mean"),
    ("High ALT (teen cutoffs)", "screening", "WTMEC2YR", "HIGH_ALT", 0, "pct"),
    ("Total sugar, average (diet sample)", "diet", "WTDRD1", "DR1TSUGR", 0, "mean"),
]
UNITS = {"LUXCAPM": " dB/m", "LBXSATSI": " U/L", "DR1TSUGR": " g/day"}


def table_a4(samples=None):
    if samples is None:
        samples = data_cutting()
    rows = []
    for label, sample, weight, col, decimals, kind in ROWS:
        df = add_variables(samples[sample])
        df["BOY"] = df["BOY"].astype(float)
        df["HISPANIC"] = (df["RACE"] == "Hispanic").astype(float)
        df["OBESE"] = (df["WEIGHT_GROUP"] == "Obesity").astype(float)
        mean, _ = weighted_mean_sd(df[col], df[weight])
        value = f"{100 * mean:.{decimals}f}%" if kind == "pct" else f"{mean:.{decimals}f}{UNITS.get(col, '')}"
        rows.append([label, value])
    n = len(samples["screening"])
    return pd.DataFrame(rows, columns=[f"Screening sample, n = {n} (weighted)", "Value"])


if __name__ == "__main__":
    table = table_a4()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)
    print(table.to_string(index=False))
    print(f"\nWrote {OUTPUT.relative_to(ROOT)}")
