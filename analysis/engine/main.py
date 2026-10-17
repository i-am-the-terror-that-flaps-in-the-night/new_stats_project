import pandas as pd

from analysis.paths import DATA_DIR, ROOT
from analysis.utilities.cleaning import data_cutting

OUTPUT = DATA_DIR / "cleaned" / "nhanes_J_cleaned.xlsx"

# All analyses use data/merged/nhanes_J.parquet, cleaned and cut into study samples
samples = data_cutting()

scan = samples["scan"]
screening = samples["screening"]
diet = samples["diet"]
fasting = samples["fasting"]

# Write each study sample to its own sheet
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with pd.ExcelWriter(OUTPUT) as writer:
    for name, sample in samples.items():
        sample.to_excel(writer, sheet_name=name, index=False)

print(f"Wrote {OUTPUT.relative_to(ROOT)}")
