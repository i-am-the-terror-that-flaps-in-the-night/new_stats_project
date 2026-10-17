from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# NHANES "refused" / "don't know" codes for the activity questions
PAQ_MISSING_CODES = [77, 99]


def clean(df):
    # read_sas turns true zeros into tiny floats like 5.4e-79, so put them back to 0
    num = df.select_dtypes("number").columns
    df[num] = df[num].mask(df[num].abs() < 1e-70, 0)

    # Refused / don't know are not real answers, and 8 means "doesn't watch TV /
    # doesn't use a computer", so count it as 0 hours (not 8)
    for col in ["PAQ710", "PAQ715"]:
        df[col] = df[col].mask(df[col].isin(PAQ_MISSING_CODES)).replace({8: 0})

    return df


def data_cutting(age_min=12, age_max=17, min_valid_scans=10, max_iqr_ratio=30):
    df = pd.read_parquet(ROOT / "data" / "merged" / "nhanes_J.parquet")
    df = clean(df)

    # Records how many people are left after each step
    flow = [("All NHANES 2017-2018", len(df))]

    def step(label, data):
        flow.append((label, len(data)))
        return data

    # Came to the mobile clinic (MEC) for the exam
    df = step("Examined at mobile clinic", df[df["RIDSTATR"] == 2])

    # Age cut
    df = step(f"Age {age_min}-{age_max}", df[df["RIDAGEYR"].between(age_min, age_max)])

    # Hepatitis cut: only drop confirmed positives, keep people who weren't tested
    hep_positive = (
        (df["LBXHBC"] == 1)    # Hep B core antibody (past/current infection)
        | (df["LBDHBG"] == 1)  # Hep B surface antigen (active infection)
        | (df["LBDHCI"].isin([1, 4]))  # Hep C antibody confirmed positive (4 = HCV RNA positive)
        | (df["LBXHCR"] == 1)  # Hep C RNA (active infection)
    )
    df = step("No hepatitis B or C", df[~hep_positive])

    # Scan sample: complete FibroScan with enough good, consistent readings.
    # Per the LUX_J codebook, "complete" already means fasted >= 3 hrs, >= 10 valid
    # readings and IQR/median < 30%; the extra checks just make the rule explicit.
    scan = df[
        (df["LUAXSTAT"] == 1)
        & (df["LUANMVGP"] >= min_valid_scans)
        & (df["LUXSIQRM"] < max_iqr_ratio)
    ]
    scan = step("Scan sample", scan)

    # Screening sample: scan sample with ALT, BMI and waist
    screening = step(
        "Screening sample (scan + ALT, BMI, waist)",
        scan.dropna(subset=["LBXSATSI", "BMXBMI", "BMXWAIST"]),
    )

    # Diet sample: scan sample with a reliable day 1 food recall
    diet = scan[(scan["DR1DRSTZ"] == 1) & scan["DR1TSUGR"].notna()]
    diet = step("Diet sample (scan + reliable day 1 recall)", diet)

    # Fasting sample: diet sample in the morning fasting group with insulin and glucose
    fasting = diet[(diet["WTSAF2YR"] > 0)].dropna(subset=["LBXIN", "LBXGLU"])
    fasting = step("Fasting sample (diet + insulin, glucose)", fasting)

    for label, n in flow:
        print(f"{label:<45} {n:>6,}")

    return {
        "scan": scan.copy(),
        "screening": screening.copy(),
        "diet": diet.copy(),
        "fasting": fasting.copy(),
    }


if __name__ == "__main__":
    data_cutting()
