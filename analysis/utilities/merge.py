from pathlib import Path

import pandas as pd

# Sets path so script can be run from anywhere
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

PARQUET = DATA_DIR / "parquet"
OUT = DATA_DIR / "merged"
OUT.mkdir(exist_ok=True)

files = sorted(PARQUET.glob("*.parquet"))
demo_file = PARQUET / "DEMO_J.parquet"
files.remove(demo_file)

df = pd.read_parquet(demo_file)

for f in files:
    other = pd.read_parquet(f)
    # Drop duplicate columns aside from SEQN (Identification number)
    duplicates = [c for c in other.columns if c in df.columns and c != "SEQN"]
    other = other.drop(columns=duplicates)
    df = pd.merge(df, other, on="SEQN", how="left", validate="one_to_one")
    print(f.name, "-->", df.shape)

df.to_parquet(OUT / "nhanes_J.parquet", index=False)
df.to_csv(OUT / "nhanes_J.csv", index=False)
df.to_excel(OUT / "nhanes_J.xlsx", index=False)

print("\nComplete.")
