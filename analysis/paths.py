from pathlib import Path

# Project folder paths, worked out from this file's location so they are the
# same no matter which folder code is run from (scripts, notebooks, PyCharm)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# Where results are written
OUTPUT_DIR = ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"  # figure_*.svg
TABLES_DIR = OUTPUT_DIR / "tables"    # table_*.csv
CLEANED_DIR = OUTPUT_DIR / "cleaned"  # study samples written by engine/main.py
