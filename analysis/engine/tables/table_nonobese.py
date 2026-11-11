import pandas as pd

from analysis.engine.figures.figure_A8 import nonobese_comparison
from analysis.engine.tables.table_B1 import format_p
from analysis.paths import ROOT, TABLES_DIR

OUTPUT = TABLES_DIR / "table_nonobese.csv"

# "Non-obese teens: the full comparison", the unnumbered table in section B11.
# Figure A8 draws most of these rows; this is every row, in the doc's order.
# Rows: label, column, decimals, which measurement it comes from (as in A8).
# Male and Hispanic are yes/no columns, shown as weighted %.
ROWS = [
    ("Age, years", "RIDAGEYR", 1, "exam"),
    ("Male, %", "BOY", 0, "exam"),
    ("Hispanic, %", "HISPANIC", 0, "exam"),
    ("BMI z-score", "BMI_Z", 2, "exam"),
    ("Waist-to-height ratio", "WHTR", 3, "exam"),
    ("Visceral fat, g", "DXXVFATM", 0, "exam"),
    ("Subcutaneous abdominal fat, g", "DXXSATM", 0, "exam"),
    ("Android (belly) fat, %", "DXXAPFAT", 1, "exam"),
    ("Android/gynoid ratio", "DXXAGRAT", 2, "exam"),
    ("ALT, U/L", "LBXSATSI", 1, "exam"),
    ("Uric acid, mg/dL", "LBXSUA", 2, "exam"),
    ("HOMA-IR", "HOMA_IR", 2, "fasting"),
    ("Fasting insulin, µU/mL", "LBXIN", 1, "fasting"),
    ("Trig/HDL ratio", "TG_HDL", 2, "fasting"),
    ("HbA1c, %", "LBXGH", 2, "fasting"),
    ("Total sugar, g/day", "DR1TSUGR", 0, "diet"),
]
PERCENT = {"BOY", "HISPANIC"}


def table_nonobese(samples=None):
    results, _, _ = nonobese_comparison(samples, ROWS)
    rows = []
    for (label, col, decimals, _), (_, r) in zip(ROWS, results.iterrows()):
        cells = []
        for side in ("with", "without"):
            value = f"{100 * r[side]:.0f}%" if col in PERCENT else f"{r[side]:.{decimals}f}"
            cells.append(f"{value} (n={int(r[f'n_{side}'])})")
        rows.append([label, *cells, format_p(r["p"])])
    return pd.DataFrame(rows, columns=["Among teens WITHOUT obesity", "With liver fat", "Without liver fat", "p"])


if __name__ == "__main__":
    table = table_nonobese()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)
    print(table.to_string(index=False))
    print(f"\nWrote {OUTPUT.relative_to(ROOT)}")
