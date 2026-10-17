from pathlib import Path

# Project folder paths, worked out from this file's location so they are the
# same no matter which folder code is run from (scripts, notebooks, PyCharm)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
