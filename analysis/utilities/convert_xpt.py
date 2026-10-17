"""
Convert every .xpt file in data/ into parquet, csv, and Excel files,
then move the original .xpt into the archive.

Resulting layout:
    data/parquet/        <- for analysis
    data/excel/          <- for viewing in Excel
    data/archive/csv/    <- plain csv copies
    data/archive/xpt/    <- original .xpt files (moved here after converting)

"""

import shutil

import pandas as pd

from analysis.paths import DATA_DIR

PARQUET_DIR = DATA_DIR / "parquet"
EXCEL_DIR = DATA_DIR / "excel"
CSV_DIR = DATA_DIR / "archive" / "csv"
XPT_ARCHIVE_DIR = DATA_DIR / "archive" / "xpt"

# Make the output folders if they don't exist yet
PARQUET_DIR.mkdir(parents=True, exist_ok=True)
EXCEL_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)
XPT_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

xpt_files = sorted(DATA_DIR.glob("*.xpt"))

if not xpt_files:
    print(f"No .xpt files found in {DATA_DIR}")

for xpt_path in xpt_files:
    name = xpt_path.stem  # e.g. "DEMO_J"
    print(f"Converting {xpt_path.name} ...")

    df = pd.read_sas(xpt_path, format="xport", encoding="utf-8")

    df.to_parquet(PARQUET_DIR / f"{name}.parquet", index=False)
    df.to_csv(CSV_DIR / f"{name}.csv", index=False)
    df.to_excel(EXCEL_DIR / f"{name}.xlsx", index=False)

    # Only move the original once all three files were written successfully
    shutil.move(xpt_path, XPT_ARCHIVE_DIR / xpt_path.name)

print("Done.")
