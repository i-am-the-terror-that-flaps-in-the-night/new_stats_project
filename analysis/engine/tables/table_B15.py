import numpy as np
import pandas as pd

from analysis.engine.tables.table_B1 import format_p
from analysis.engine.tables.table_B2 import WEIGHT
from analysis.engine.tables.table_B14 import dxa_variables
from analysis.engine.tables.table_B6 import OUTCOME
from analysis.paths import ROOT, TABLES_DIR
from analysis.utilities.survey import design_psus, logistic_regression, regression, weighted_mean_sd

OUTPUT = TABLES_DIR / "table_B15.csv"

# Small models in teens WITHOUT obesity who have a valid DXA scan: visceral fat
# (per SD, as in Table B14) next to one other measure. The odds models hold sex
# and Hispanic ethnicity steady; the CAP models also hold age steady (that is
# what reproduces the doc, though its note mentions only sex and Hispanic).
ODDS_ADJUST = ["BOY", "HISPANIC"]
CAP_ADJUST = ["BOY", "HISPANIC", "RIDAGEYR"]

# Each model: name, second measure, its label, and the weight (HOMA-IR comes
# from the fasting subsample, so that model uses the fasting weight)
MODELS = [
    ("Visceral fat + waist-to-height", "WHTR_SD", "Waist-to-height (per SD)", WEIGHT),
    ("Visceral fat + HOMA-IR", "LOG_HOMA_IR", "HOMA-IR (per log unit)", "WTSAF2YR"),
    ("Visceral fat + uric acid", "LBXSUA", "Uric acid (per mg/dL)", WEIGHT),
]


def p_text(p):
    p = format_p(p)
    return f"p {p if p.startswith('<') else '= ' + p}"


def table_b15(samples=None):
    psus = design_psus()
    df, dxa = dxa_variables(samples)
    # Waist-to-height per SD, standardized the same way as the Table B14 measures
    mean, sd = weighted_mean_sd(df.loc[dxa, "WHTR"], df.loc[dxa, WEIGHT])
    df["WHTR_SD"] = ((df["WHTR"] - mean) / sd).where(dxa)

    # Only non-obese teens count; everyone stays in for the variance
    domain = dxa & (df["WEIGHT_GROUP"] != "Obesity")
    df["FAT_IN"] = df["LIVER_FAT"].where(domain)
    df["CAP_IN"] = df[OUTCOME].where(domain)

    rows = []
    for name, col, label, weight in MODELS:
        shown = [("VISCERAL_SD", "Visceral fat"), (col, label)]
        odds, fit = logistic_regression(df, "FAT_IN", ["VISCERAL_SD", col, *ODDS_ADJUST], weight, psus)
        cap, _ = regression(df, "CAP_IN", ["VISCERAL_SD", col, *CAP_ADJUST], weight, psus)
        odds_text = []
        for c, text in shown:
            est, low, high = np.exp(odds.loc[c, ["estimate", "low", "high"]])
            odds_text.append(f"{text}: {est:.2f} [{low:.2f}, {high:.2f}], {p_text(odds.loc[c, 'p'])}")
        cap_text = [f"{text}: {cap.loc[c, 'estimate']:.0f}, {p_text(cap.loc[c, 'p'])}" for c, text in shown]
        rows.append([name, f"{fit['n']} ({fit['events']})", "; ".join(odds_text), "; ".join(cap_text)])
    return pd.DataFrame(rows, columns=[
        "Small model (non-obese teens only)",
        "Teens (with liver fat)",
        "Odds of liver fat [95% CI]",
        "Change in CAP (dB/m)",
    ])


if __name__ == "__main__":
    table = table_b15()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)
    with pd.option_context("display.width", 300, "display.max_colwidth", 120):
        print(table.to_string(index=False))
    print(f"\nWrote {OUTPUT.relative_to(ROOT)}")
