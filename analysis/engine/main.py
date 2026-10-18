import pandas as pd

from analysis.paths import ROOT
from analysis.utilities.cleaning import data_cutting

OUTPUT_DIR = ROOT / "output" / "cleaned"
EXCEL = OUTPUT_DIR / "nhanes_J_cleaned.xlsx"

# All analyses use data/merged/nhanes_J.parquet, cleaned and cut into study samples
samples = data_cutting()

scan = samples["scan"]
screening = samples["screening"]
diet = samples["diet"]
fasting = samples["fasting"]

# Write each study sample to its own Excel sheet, and to its own csv
# (csv has no sheets, so one file per sample)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
with pd.ExcelWriter(EXCEL) as writer:
    for name, sample in samples.items():
        sample.to_excel(writer, sheet_name=name, index=False)
print(f"Wrote {EXCEL.relative_to(ROOT)}")

for name, sample in samples.items():
    csv = OUTPUT_DIR / f"nhanes_J_{name}.csv"
    sample.to_csv(csv, index=False)
    print(f"Wrote {csv.relative_to(ROOT)}")
