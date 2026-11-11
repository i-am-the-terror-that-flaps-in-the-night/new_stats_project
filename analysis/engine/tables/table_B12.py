import pandas as pd

from analysis.engine.tables.table_B1 import format_p
from analysis.engine.tables.table_B6 import MODELS, OUTCOME, model_variables
from analysis.paths import ROOT, TABLES_DIR
from analysis.utilities.cleaning import data_cutting
from analysis.utilities.survey import design_psus, regression

OUTPUT = TABLES_DIR / "table_B12.csv"

URIC_ACID = "LBXSUA"  # mg/dL

# Both questions use the diet sample and diet weight, holding Model A's other
# factors (everything but sugar) steady
_, SAMPLE, WEIGHT, MODEL_A = MODELS[0]
ADJUST = [col for col in MODEL_A if col != "SUGAR"]


def table_b12(samples=None):
    if samples is None:
        samples = data_cutting()
    psus = design_psus()
    df = model_variables(samples[SAMPLE])

    # 1. Does sugar predict uric acid? (mg/dL per 50 g/day)
    sugar, sugar_fit = regression(df, URIC_ACID, ["SUGAR", *ADJUST], WEIGHT, psus)
    # 2. Does uric acid predict liver fat? (dB/m per mg/dL)
    uric, uric_fit = regression(df, OUTCOME, [URIC_ACID, *ADJUST], WEIGHT, psus)

    s_est, s_low, s_high, s_p = sugar.loc["SUGAR"]
    u_est, u_low, u_high, u_p = uric.loc[URIC_ACID]
    rows = [
        ["Does sugar predict uric acid? (mg/dL per 50 g)", sugar_fit["n"],
         f"{s_est:+.2f} [{s_low:+.2f}, {s_high:+.2f}]", format_p(s_p), f"{s_est:+.2f} mg/dL"],
        ["Does uric acid predict liver fat? (dB/m per mg/dL)", uric_fit["n"],
         f"{u_est:.1f} [{u_low:.1f}, {u_high:.1f}]", format_p(u_p), "—"],
    ]
    return pd.DataFrame(rows, columns=["Question", "Teens", "Effect [95% CI]", "p", "Per 50 g sugar"])


if __name__ == "__main__":
    table = table_b12()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)
    with pd.option_context("display.width", 200):
        print(table.to_string(index=False))
    print(f"\nWrote {OUTPUT.relative_to(ROOT)}")
