import numpy as np
import pandas as pd
from scipy import stats

from analysis.paths import ROOT
from analysis.utilities.cleaning import data_cutting
from analysis.utilities.survey import (
    compare_categories,
    compare_means,
    design_psus,
    weighted_mean_sd,
)

OUTPUT = ROOT / "output" / "table_B1.csv"

# Liver fat and stiffness cutoffs used throughout the results
CAP_FAT = 248  # dB/m
STIFFNESS_HIGH = 7.4  # kPa

# Teen ALT cutoffs (U/L), above which ALT counts as high
ALT_HIGH = {1: 26, 2: 22}  # boys, girls

# CDC weight groups from BMI-for-age percentiles: <5th, 5th-<85th, 85th-<95th, >=95th
WEIGHT_GROUPS = ["Underweight", "Healthy weight", "Overweight", "Obesity"]
WEIGHT_CUTS = [-np.inf, stats.norm.ppf(0.05), stats.norm.ppf(0.85), stats.norm.ppf(0.95), np.inf]

# RIDRETH3 codes, with Mexican American (1) and other Hispanic (2) combined
RACE = {
    3: "Non-Hispanic White",
    1: "Hispanic",
    2: "Hispanic",
    4: "Non-Hispanic Black",
    6: "Non-Hispanic Asian",
    7: "Other/Multiracial",
}

# Each section: heading, which sample and weight it uses, and its rows.
# Row kinds: "mean" -> mean (SD), "pct" -> weighted % of a yes/no column,
# "cat" -> number (weighted %) for each category of a column.
SECTIONS = [
    ("Screening sample", "screening", "WTMEC2YR", [
        ("mean", "Age, years", "RIDAGEYR", 1),
        ("mean", "BMI z-score", "BMI_Z", 2),
        ("mean", "Waist-to-height ratio", "WHTR", 3),
        ("cat", "WEIGHT_GROUP", WEIGHT_GROUPS),
        ("cat", "RACE", list(dict.fromkeys(RACE.values()))),
        ("cat", "XL_WAND", ["XL wand used"]),
    ]),
    ("FibroScan", "screening", "WTMEC2YR", [
        ("mean", "Liver fat, CAP (dB/m)", "LUXCAPM", 0),
        ("pct", f"Liver fat present (CAP ≥{CAP_FAT})", "LIVER_FAT"),
        ("mean", "Liver stiffness (kPa)", "LUXSMED", 1),
        ("pct", f"High stiffness (≥{STIFFNESS_HIGH} kPa)", "HIGH_STIFFNESS"),
    ]),
    ("Blood tests", "screening", "WTMEC2YR", [
        ("mean", "ALT, U/L", "LBXSATSI", 1),
        ("pct", "High ALT (teen cutoffs)", "HIGH_ALT"),
        ("mean", "Uric acid, mg/dL", "LBXSUA", 1),
    ]),
    ("Diet and lifestyle (diet sample)", "diet", "WTDRD1", [
        ("mean", "Total sugar, grams/day", "DR1TSUGR", 1),
        ("mean", "Calories/day", "DR1TKCAL", 0),
        ("mean", "Days active ≥60 min, past week", "PAQ706", 1),
        ("mean", "Screen time, hours/day", "SCREEN_TIME", 1),
    ]),
    ("Fasting blood tests (fasting sample)", "fasting", "WTSAF2YR", [
        ("mean", "HOMA-IR", "HOMA_IR", 2),
        ("mean", "Fasting insulin, µU/mL", "LBXIN", 1),
        ("mean", "Trig/HDL ratio", "TG_HDL", 2),
        ("mean", "HbA1c, %", "LBXGH", 2),
    ]),
]


def add_variables(df):
    # Derived variables used in the table
    df = df.copy()
    boy = df["RIAGENDR"] == 1
    df["WHTR"] = df["BMXWAIST"] / df["BMXHT"]
    df["WEIGHT_GROUP"] = pd.cut(df["BMI_Z"], WEIGHT_CUTS, right=False, labels=WEIGHT_GROUPS)
    df["RACE"] = df["RIDRETH3"].map(RACE)
    # Probe name is "M" or "XL"; blank means not recorded
    probe = df["LUAPNME"].replace("", np.nan)
    df["XL_WAND"] = probe.map({"XL": "XL wand used", "M": "M wand"})
    df["LIVER_FAT"] = (df["LUXCAPM"] >= CAP_FAT).astype(float).where(df["LUXCAPM"].notna())
    df["HIGH_STIFFNESS"] = (df["LUXSMED"] >= STIFFNESS_HIGH).astype(float).where(df["LUXSMED"].notna())
    alt_cutoff = df["RIAGENDR"].map(ALT_HIGH)
    df["HIGH_ALT"] = (df["LBXSATSI"] > alt_cutoff).astype(float).where(df["LBXSATSI"].notna())
    # TV + computer hours (each 0-5, where 5 means 5 or more)
    df["SCREEN_TIME"] = df["PAQ710"] + df["PAQ715"]
    # HOMA-IR = fasting glucose (mg/dL) x fasting insulin (µU/mL) / 405
    df["HOMA_IR"] = df["LBXGLU"] * df["LBXIN"] / 405
    df["TG_HDL"] = df["LBXTR"] / df["LBDHDD"]
    df["BOY"] = boy
    return df


def format_p(p):
    if pd.isna(p):
        return ""
    return "< 0.001" if p < 0.001 else f"{p:.2f}" if p >= 0.1 else f"{p:.3f}"


def mean_row(df, label, col, decimals, weight, psus):
    cells = [f"{label}, mean (SD)"]
    for part in [df, df[df["BOY"]], df[~df["BOY"]]]:
        mean, sd = weighted_mean_sd(part[col], part[weight])
        cells.append(f"{mean:.{decimals}f} ({sd:.{decimals}f})")
    cells.append(format_p(compare_means(df, col, weight, df["BOY"], psus)))
    return cells


def pct_row(df, label, col, weight, psus):
    cells = [f"{label}, weighted %"]
    for part in [df, df[df["BOY"]], df[~df["BOY"]]]:
        mean, _ = weighted_mean_sd(part[col], part[weight])
        cells.append(f"{100 * mean:.1f}%")
    cells.append(format_p(compare_means(df, col, weight, df["BOY"], psus)))
    return cells


def category_rows(df, col, shown, weight, psus):
    # One row per category; the p-value tests all categories at once, on the first row
    p = compare_categories(df, col, weight, df["BOY"], psus)
    rows = []
    for level in shown:
        cells = [f"  {level}, number (weighted %)"]
        for part in [df, df[df["BOY"]], df[~df["BOY"]]]:
            known = part[col].notna()
            is_level = part[col] == level
            pct = 100 * part.loc[is_level, weight].sum() / part.loc[known, weight].sum()
            cells.append(f"{is_level.sum()} ({pct:.1f}%)")
        cells.append(format_p(p) if not rows else "")
        rows.append(cells)
    return rows


def table_b1(samples=None):
    if samples is None:
        samples = data_cutting()
    psus = design_psus()

    rows = []
    for heading, sample, weight, specs in SECTIONS:
        df = add_variables(samples[sample])
        n = f"n = {len(df)}"
        rows.append([f"{heading} ({n})", "", "", "", ""])
        for kind, *spec in specs:
            if kind != "cat" and spec[1] not in df:
                print(f"Skipping '{spec[0]}': column {spec[1]} is not in the data")
                continue
            if kind == "mean":
                rows.append(mean_row(df, *spec, weight, psus))
            elif kind == "pct":
                rows.append(pct_row(df, *spec, weight, psus))
            else:
                rows.extend(category_rows(df, *spec, weight, psus))

    screening = samples["screening"]
    boys = (screening["RIAGENDR"] == 1).sum()
    header = ["Characteristic", "Everyone", f"Boys (n = {boys})", f"Girls (n = {len(screening) - boys})",
              "p (boys vs. girls)"]
    return pd.DataFrame(rows, columns=header)


if __name__ == "__main__":
    table = table_b1()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(table.to_string(index=False))
    print(f"\nWrote {OUTPUT.relative_to(ROOT)}")
