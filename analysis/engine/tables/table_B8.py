import numpy as np
import pandas as pd

from analysis.engine.tables.table_B1 import format_p
from analysis.engine.tables.table_B6 import MODELS, OUTCOME, model_variables
from analysis.paths import ROOT, TABLES_DIR
from analysis.utilities.cleaning import data_cutting
from analysis.utilities.survey import design_psus, regression

OUTPUT = TABLES_DIR / "table_B8.csv"

_, _, _, MODEL_A = MODELS[0]


def fit_teens(df, predictors):
    # Teens who are in a model's fit: outcome and every predictor present
    return df[[OUTCOME, *predictors]].notna().all(axis=1)


def highest_vif(df, predictors):
    # Variance inflation factor of each predictor: 1 / (1 - R²) from regressing
    # it on all the others. Above 5 means two predictors overlap too much to
    # tell apart. Plain (unweighted): it describes the teens the model sees.
    X = df[predictors].astype(float)
    vifs = []
    for col in predictors:
        others = np.column_stack([np.ones(len(X)), X.drop(columns=col)])
        beta, *_ = np.linalg.lstsq(others, X[col], rcond=None)
        resid = X[col] - others @ beta
        r2 = 1 - (resid**2).sum() / ((X[col] - X[col].mean()) ** 2).sum()
        vifs.append(1 / (1 - r2))
    return max(vifs)


def model_fit(samples=None):
    # How well each model in MODELS works: one dict per model
    if samples is None:
        samples = data_cutting()
    psus = design_psus()
    fits = {}
    for name, sample, weight, predictors in MODELS:
        df = model_variables(samples[sample])
        _, fit = regression(df, OUTCOME, predictors, weight, psus)
        n, k = fit["n"], len(predictors)

        # Model A refit on this model's teens and weight, so the gain in R²
        # comes from the added predictors and not from a different set of teens
        same = df.assign(**{OUTCOME: df[OUTCOME].where(fit_teens(df, predictors))})
        _, base = regression(same, OUTCOME, MODEL_A, weight, psus)

        fits[name] = {
            "n": n,
            "r2": fit["r2"],
            "adj_r2": 1 - (1 - fit["r2"]) * (n - 1) / (n - k - 1),
            "gain": None if predictors == MODEL_A else fit["r2"] - base["r2"],
            "p": fit["p"],
            "vif": highest_vif(df[fit_teens(df, predictors)], predictors),
        }
    return fits


def table_b8(samples=None):
    fits = model_fit(samples)
    rows = [
        ("Number of teens", lambda f: f["n"]),
        ("Share of liver-fat variation explained (R²)", lambda f: f"{f['r2']:.2f}"),
        ("Adjusted R²", lambda f: f"{f['adj_r2']:.2f}"),
        ("Gain over Model A on the same teens", lambda f: "—" if f["gain"] is None else f"{f['gain']:+.2f}"),
        ("Overall test p-value", lambda f: format_p(f["p"])),
        ("Highest overlap between predictors (VIF; should be under 5)", lambda f: f"{f['vif']:.2f}"),
    ]
    return pd.DataFrame(
        [[label, *[cell(fit) for fit in fits.values()]] for label, cell in rows],
        columns=["How well the models work", *fits],
    )


if __name__ == "__main__":
    table = table_b8()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)
    with pd.option_context("display.width", 250):
        print(table.to_string(index=False))
    print(f"\nWrote {OUTPUT.relative_to(ROOT)}")
