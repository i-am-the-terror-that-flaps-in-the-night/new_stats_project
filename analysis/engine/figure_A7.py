import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.engine.table_B1 import format_p
from analysis.engine.table_B6 import MODELS, fit_models
from analysis.paths import ROOT
from analysis.utilities.cleaning import data_cutting

OUTPUT = ROOT / "output" / "figure_A7.svg"

# The final model: Model B with BMI, the one with the promised sugar test
FINAL = MODELS[2][0]

# Rows, top to bottom: label, coefficient, and a factor to rescale it by.
# HOMA-IR is in the model as a log, so one log unit times ln 2 is the change
# per doubling. Other race groups and the XL wand are in the model, not drawn.
ROWS = [
    ("Total sugar, per extra 50 g/day", "SUGAR", 1),
    ("Calories, per extra 500/day", "KCAL", 1),
    ("Age, per year older", "RIDAGEYR", 1),
    ("Boy vs. girl", "BOY", 1),
    ("Hispanic vs. Non-Hispanic White", "HISPANIC", 1),
    ("Insulin resistance (HOMA-IR), per doubling", "LOG_HOMA_IR", np.log(2)),
    ("Body size (BMI z-score), per 1 SD", "BMI_Z", 1),
]
PRIMARY = 0  # index of the promised sugar test in ROWS

BLUE, DARK_BLUE, RED = "#2e75b6", "#1f4e79", "#c0392b"


def signed(x):
    # Whole numbers with a sign, but one decimal for values that round to 0,
    # so -0.4 shows as -0.4 rather than a confusing -0
    return f"{x:+.1f}" if round(x) == 0 else f"{x:+.0f}"


def final_model(samples=None):
    # The drawn rows of the final model, rescaled: estimate, 95% CI and p
    if samples is None:
        samples = data_cutting()
    coefs, fit = fit_models(samples)[FINAL]
    rows = []
    for label, col, scale in ROWS:
        est, low, high, p = coefs.loc[col]
        rows.append({"label": label, "estimate": est * scale, "low": low * scale, "high": high * scale, "p": p})
    return pd.DataFrame(rows).set_index("label"), fit["n"]


def figure_a7(samples=None):
    effects, n = final_model(samples)

    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=200)
    y = range(len(effects))[::-1]
    for i, (row, (label, e)) in enumerate(zip(y, effects.iterrows())):
        colour = RED if e["p"] < 0.05 else DARK_BLUE if i == PRIMARY else BLUE
        ax.errorbar(e["estimate"], row, xerr=[[e["estimate"] - e["low"]], [e["high"] - e["estimate"]]],
                    fmt="D" if i == PRIMARY else "o", markersize=7 if i == PRIMARY else 5,
                    color=colour, ecolor=colour, elinewidth=1.3, capsize=3)
        p = format_p(e["p"])
        ax.text(1.02, row, f"{signed(e['estimate'])}  ({signed(e['low'])} to {signed(e['high'])})   "
                f"p {p if p.startswith('<') else '= ' + p}",
                transform=ax.get_yaxis_transform(), va="center", fontsize=8)

    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_yticks(list(y), effects.index, fontsize=8.5)
    ax.set_xlabel("Change in liver fat (CAP, dB/m), with everything else held steady. Red = p < 0.05",
                  fontsize=9)
    ax.set_title(f"What goes with liver fat: the final model (n = {n} teens)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return fig, effects, n


if __name__ == "__main__":
    fig, effects, n = figure_a7()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    print(f"n = {n}")
    print(effects.round(3).to_string())
    print(f"\nWrote {OUTPUT.relative_to(ROOT)}")
