import numpy as np
import pandas as pd

from analysis.engine.tables.table_B1 import RACE, add_variables, format_p
from analysis.paths import ROOT, TABLES_DIR
from analysis.utilities.cleaning import data_cutting
from analysis.utilities.survey import design_psus, regression

OUTPUT = TABLES_DIR / "table_B6.csv"

OUTCOME = "LUXCAPM"  # liver fat, CAP (dB/m)

# Units the effects are reported in
SUGAR_UNIT = 50  # g/day
KCAL_UNIT = 500  # calories/day

# Race/ethnicity dummies, each compared with Non-Hispanic White
RACE_DUMMIES = {
    "Hispanic": "HISPANIC",
    "Non-Hispanic Black": "NH_BLACK",
    "Non-Hispanic Asian": "NH_ASIAN",
    "Other/Multiracial": "OTHER_RACE",
}

# Every model holds these steady; sugar is the one test promised in advance
BASE = ["SUGAR", "KCAL", "RIDAGEYR", "BOY", *RACE_DUMMIES.values(), "XL"]

# Each model: name, which sample and weight it uses, and its predictors
MODELS = [
    ("Model A (lifestyle)", "diet", "WTDRD1", BASE),
    ("Model B (adds HOMA-IR)", "fasting", "WTSAF2YR", [*BASE, "LOG_HOMA_IR"]),
    ("Model B with BMI ★ (the poster model)", "fasting", "WTSAF2YR", [*BASE, "LOG_HOMA_IR", "BMI_Z"]),
]

# Table rows, top to bottom: section headings (None) and predictor labels
ROWS = [
    ("Total sugar (per 50 g/day)", "SUGAR"),
    ("Calories (per 500/day)", "KCAL"),
    ("Age (per year)", "RIDAGEYR"),
    ("Boy (vs. girl)", "BOY"),
    ("Race/ethnicity, compared with Non-Hispanic White", None),
    *[(label, col) for label, col in RACE_DUMMIES.items()],
    ("Scan, metabolism, body size", None),
    ("XL wand (vs. regular)", "XL"),
    ("HOMA-IR (per log unit)", "LOG_HOMA_IR"),
    ("BMI z-score (per 1 SD)", "BMI_Z"),
]


def model_variables(df, two_day=False):
    # Predictors in the units the table reports them in. With two_day, sugar and
    # calories are the average of both recalls, for teens whose day 2 recall is
    # reliable (everyone else is set to missing, so drops out of the fit).
    df = add_variables(df)
    sugar, kcal = df["DR1TSUGR"], df["DR1TKCAL"]
    if two_day:
        day2 = df["DR2DRSTZ"] == 1
        sugar = ((sugar + df["DR2TSUGR"]) / 2).where(day2)
        kcal = ((kcal + df["DR2TKCAL"]) / 2).where(day2)
    df["SUGAR"] = sugar / SUGAR_UNIT
    df["KCAL"] = kcal / KCAL_UNIT
    df["BOY"] = df["BOY"].astype(float)
    for race, col in RACE_DUMMIES.items():
        df[col] = (df["RACE"] == race).astype(float).where(df["RACE"].notna())
    df["XL"] = (df["XL_WAND"] == "XL wand used").astype(float).where(df["XL_WAND"].notna())
    df["LOG_HOMA_IR"] = np.log(df["HOMA_IR"])
    return df


def fit_models(samples=None, two_day=False):
    # Fit every model in MODELS. Returns {name: (coefficients, fit)}.
    # With two_day, Model A uses the two-day diet weight; the fasting models keep
    # the fasting weight, since the fasting sample is the smaller subsample.
    if samples is None:
        samples = data_cutting()
    psus = design_psus()
    fits = {}
    for name, sample, weight, predictors in MODELS:
        if two_day and weight == "WTDRD1":
            weight = "WTDR2D"
        df = model_variables(samples[sample], two_day)
        fits[name] = regression(df, OUTCOME, predictors, weight, psus)
    return fits


def cell(coefs, col):
    if col not in coefs.index:
        return "—"
    est, low, high, p = coefs.loc[col]
    p = format_p(p)
    return f"{est:.1f} [{low:.1f}, {high:.1f}], p {p if p.startswith('<') else '= ' + p}"


def table_b6(samples=None):
    fits = fit_models(samples)
    rows = []
    for label, col in ROWS:
        if col is None:
            rows.append([label, *[""] * len(fits)])
        else:
            rows.append([label, *[cell(coefs, col) for coefs, _ in fits.values()]])
    rows.append(["Number of teens", *[fit["n"] for _, fit in fits.values()]])
    rows.append(["R²", *[f"{fit['r2']:.2f}" for _, fit in fits.values()]])
    return pd.DataFrame(rows, columns=["Change in liver fat (CAP, dB/m) per unit", *fits])


if __name__ == "__main__":
    table = table_b6()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 40):
        print(table.to_string(index=False))
    print(f"\nWrote {OUTPUT.relative_to(ROOT)}")
