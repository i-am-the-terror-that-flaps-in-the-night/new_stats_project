import matplotlib.pyplot as plt
import pandas as pd

from analysis.engine.table_B1 import format_p
from analysis.engine.table_B6 import MODELS, fit_models
from analysis.paths import FIGURES_DIR, ROOT
from analysis.utilities.cleaning import data_cutting

OUTPUT = FIGURES_DIR / "figure_B5.svg"

# Rows, top to bottom: label, model name (from Table B6), and whether sugar and
# calories are the two-day average (the S1 check) instead of day 1 only
MODEL_A, MODEL_B, MODEL_B_BMI = (name for name, *_ in MODELS)
ROWS = [
    ("Model A (lifestyle), one-day recall", MODEL_A, False),
    ("Model B without BMI", MODEL_B, False),
    ("Model B with BMI ★ (primary)", MODEL_B_BMI, False),
    ("S1: two-day average, Model A", MODEL_A, True),
    ("S1: two-day average, Model B with BMI", MODEL_B_BMI, True),
]
PRIMARY = 2  # index of the promised primary test in ROWS

# Blue for most rows, dark blue for the primary test, red where p < 0.05
BLUE, DARK_BLUE, RED = "#2e75b6", "#1f4e79", "#c0392b"


def sugar_effects(samples=None):
    # The sugar row of every model: estimate, 95% CI, p and n, one row per ROWS
    if samples is None:
        samples = data_cutting()
    fits = {two_day: fit_models(samples, two_day) for two_day in (False, True)}
    rows = []
    for label, model, two_day in ROWS:
        coefs, fit = fits[two_day][model]
        rows.append({"label": label, **coefs.loc["SUGAR"], "n": fit["n"]})
    return pd.DataFrame(rows).set_index("label")


def figure_b5(samples=None):
    effects = sugar_effects(samples)

    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=200)
    # Rows run top to bottom in ROWS order
    y = range(len(effects))[::-1]
    for i, (row, (label, e)) in enumerate(zip(y, effects.iterrows())):
        colour = RED if e["p"] < 0.05 else DARK_BLUE if i == PRIMARY else BLUE
        ax.errorbar(e["estimate"], row, xerr=[[e["estimate"] - e["low"]], [e["high"] - e["estimate"]]],
                    fmt="D" if i == PRIMARY else "o", markersize=8 if i == PRIMARY else 5,
                    color=colour, ecolor=colour, elinewidth=1.4, capsize=3)
        p = format_p(e["p"])
        ax.text(1.02, row, f"p {p if p.startswith('<') else '= ' + p}",
                transform=ax.get_yaxis_transform(), va="center", fontsize=9)

    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    labels = [f"{label} (n = {n})" for label, n in zip(effects.index, effects["n"])]
    ax.set_yticks(list(y), labels, fontsize=10)
    ax.set_xlabel("Change in liver fat (CAP, dB/m) per extra 50 g sugar/day, 95% CI", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    # The title is the finding, worked out from the numbers
    real = (effects["p"] < 0.05).sum()
    ax.set_title(f"The sugar effect in every model: {real} of {len(effects)} reach p < 0.05", fontsize=11)

    fig.tight_layout()
    return fig, effects


if __name__ == "__main__":
    fig, effects = figure_b5()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    print(effects.round(3).to_string())
    print(f"\nWrote {OUTPUT.relative_to(ROOT)}")
