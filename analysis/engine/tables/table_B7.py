import numpy as np
import pandas as pd

from analysis.engine.tables.table_B1 import format_p
from analysis.engine.tables.table_B6 import MODELS, ROWS, model_variables
from analysis.paths import ROOT, TABLES_DIR
from analysis.utilities.cleaning import data_cutting
from analysis.utilities.survey import design_psus, logistic_regression

OUTPUT = TABLES_DIR / "table_B7.csv"

# The Table B6 models again, but for the odds of HAVING liver fat (CAP ≥248)
# instead of the CAP number itself
OUTCOME = "LIVER_FAT"


def fit_logistic_models(samples=None):
    # Fit every model in MODELS as a logistic regression.
    # Returns {name: (coefficients on the log-odds scale, fit)}.
    if samples is None:
        samples = data_cutting()
    psus = design_psus()
    return {
        name: logistic_regression(model_variables(samples[sample]), OUTCOME, predictors, weight, psus)
        for name, sample, weight, predictors in MODELS
    }


def cell(coefs, col):
    # Odds ratio [95% CI], p
    if col not in coefs.index:
        return "—"
    est, low, high = np.exp(coefs.loc[col, ["estimate", "low", "high"]])
    p = format_p(coefs.loc[col, "p"])
    return f"{est:.2f} [{low:.2f}, {high:.2f}], p {p if p.startswith('<') else '= ' + p}"


def table_b7(samples=None):
    fits = fit_logistic_models(samples)
    rows = []
    for label, col in ROWS:
        if col is None:
            rows.append([label, *[""] * len(fits)])
        else:
            rows.append([label, *[cell(coefs, col) for coefs, _ in fits.values()]])
    rows.append(["Number of teens (with liver fat)", *[f"{fit['n']} ({fit['events']})" for _, fit in fits.values()]])
    names = [name for name, *_ in MODELS]
    return pd.DataFrame(rows, columns=["Odds of HAVING liver fat (CAP ≥248) per unit", *names])


if __name__ == "__main__":
    table = table_b7()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 40):
        print(table.to_string(index=False))
    print(f"\nWrote {OUTPUT.relative_to(ROOT)}")
