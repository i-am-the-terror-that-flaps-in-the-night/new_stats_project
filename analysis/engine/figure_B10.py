import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.engine.table_B1 import CAP_FAT, format_p
from analysis.engine.table_B6 import model_variables
from analysis.paths import FIGURES_DIR, ROOT
from analysis.utilities.cleaning import data_cutting
from analysis.utilities.survey import design_psus, mean_ci, regression

OUTPUT = FIGURES_DIR / "figure_B10.svg"

WEIGHT = "WTSAF2YR"  # fasting weight: Trig/HDL and HbA1c come from the fasting sample

# One point for each of these above its sample median, plus one for being a boy
SCORE_PARTS = {
    "DR1TSUGR": "sugar",
    "SCREEN_TIME": "screen time",
    "TG_HDL": "Trig/HDL",
    "LBXGH": "HbA1c",
    "BMI_Z": "BMI z-score",
}

# The measures compared by how well they rank teens: column and bar label
MEASURES = [
    ("SCORE", "0–6 risk score\n(six factors)"),
    ("LBXSATSI", "ALT\n(blood test)"),
    ("BMI_Z", "BMI z-score\n(scale)"),
    ("WHTR", "Waist-to-height ratio\n(tape measure)"),
]

DARK_BLUE, LIGHT_BLUE, BLUE, RED = "#1f4e79", "#9dc3e6", "#2e75b6", "#c0392b"


def add_score(df):
    # The 0-6 risk score. Medians are the plain (unweighted) sample medians, as
    # the protocol set them. A teen missing any part gets no score at all, since
    # counting a missing part as 0 points would pull their score down.
    df = df.copy()
    medians = {col: df[col].median() for col in SCORE_PARTS}
    above = pd.DataFrame({col: df[col] > medians[col] for col in SCORE_PARTS})
    complete = df[list(SCORE_PARTS)].notna().all(axis=1)
    df["SCORE"] = (above.sum(axis=1) + df["BOY"]).where(complete)
    return df, medians


def weighted_auc(score, outcome, w):
    # AUC: the chance that a random teen with liver fat scores higher than a
    # random teen without (ties count half), with each pair weighted by both
    # teens' survey weights, so it describes U.S. teens rather than the sample
    has, hasnt = outcome == 1, outcome == 0
    s1, s0 = score[has].to_numpy()[:, None], score[hasnt].to_numpy()[None, :]
    pair_weight = w[has].to_numpy()[:, None] * w[hasnt].to_numpy()[None, :]
    wins = (s1 > s0) + 0.5 * (s1 == s0)
    return (wins * pair_weight).sum() / pair_weight.sum()


def risk_score(samples=None):
    # Liver fat at each score (teens, mean CAP and % with liver fat, each with a
    # 95% CI), the rise in CAP per point, and each measure's AUC
    if samples is None:
        samples = data_cutting()
    psus = design_psus()
    df, medians = add_score(model_variables(samples["fasting"]))

    rows = []
    for points in sorted(df["SCORE"].dropna().unique()):
        at = df["SCORE"] == points
        cap = mean_ci(df, "LUXCAPM", WEIGHT, at, psus)
        fat = mean_ci(df, "LIVER_FAT", WEIGHT, at, psus)
        rows.append({
            "score": int(points),
            "n": at.sum(),
            "cap": cap[0], "cap_low": cap[1], "cap_high": cap[2],
            "fat": 100 * fat[0], "fat_low": 100 * max(fat[1], 0), "fat_high": 100 * fat[2],
        })
    scores = pd.DataFrame(rows).set_index("score")

    coefs, _ = regression(df, "LUXCAPM", ["SCORE"], WEIGHT, psus)
    slope = coefs.loc["SCORE"]

    # Every AUC uses the same teens (those with all four measures), so the
    # comparison is fair
    cols = [col for col, _ in MEASURES]
    same = df.dropna(subset=[*cols, "LIVER_FAT"])
    aucs = pd.Series({col: weighted_auc(same[col], same["LIVER_FAT"], same[WEIGHT]) for col in cols})
    return scores, slope, aucs, len(same), medians


def figure_b10(samples=None):
    scores, slope, aucs, n_auc, medians = risk_score(samples)

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 3.6), dpi=200,
                                      gridspec_kw={"width_ratios": [1, 1.05]})

    # Left: mean CAP at each score, with its 95% CI as a band
    x = scores.index.to_numpy()
    left.fill_between(x, scores["cap_low"], scores["cap_high"], color=LIGHT_BLUE, alpha=0.5, linewidth=0.8)
    left.plot(x, scores["cap"], "o-", color=DARK_BLUE, linewidth=1.6, markersize=6)
    left.axhline(CAP_FAT, color=RED, linestyle="--", linewidth=1)
    left.set_xticks(x, [f"{s}\nn={n}" for s, n in zip(x, scores["n"])], fontsize=8)
    left.set_xlabel("Risk score (0–6)", fontsize=10)
    left.set_ylabel("Weighted mean CAP (dB/m)", fontsize=10)
    p = format_p(slope["p"])
    left.set_title(f"Mean liver fat by score: {slope['estimate']:+.0f} dB/m per point, "
                   f"p {p if p.startswith('<') else '= ' + p}", fontsize=10)
    left.spines[["top", "right"]].set_visible(False)

    # Right: AUC of each measure, top to bottom in MEASURES order; the best is red
    y = range(len(MEASURES))[::-1]
    best = aucs.idxmax()
    colours = [RED if col == best else BLUE for col, _ in MEASURES]
    right.barh(y, aucs, color=colours)
    for row, value in zip(y, aucs):
        right.text(value + 0.005, row, f"{value:.2f}", va="center", fontsize=9)
    right.set_yticks(list(y), [label for _, label in MEASURES], fontsize=9)
    right.set_xlim(0.5, 1.0)
    right.set_xlabel("AUC (0.5 = coin flip, 1.0 = perfect)", fontsize=10)
    right.set_title(f"How well each measure ranks teens by liver fat (n = {n_auc})", fontsize=10)
    right.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return fig, scores, slope, aucs, medians


if __name__ == "__main__":
    fig, scores, slope, aucs, medians = figure_b10()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    print("Medians: " + ", ".join(f"{SCORE_PARTS[col]} > {m:.2f}" for col, m in medians.items()))
    with pd.option_context("display.width", 200):
        print(scores.round(1).to_string())
    print(f"Per point: {slope['estimate']:+.1f} dB/m [{slope['low']:.1f}, {slope['high']:.1f}], "
          f"p {format_p(slope['p'])}")
    print("AUC: " + ", ".join(f"{col} {auc:.2f}" for col, auc in aucs.items()))
    print(f"\nWrote {OUTPUT.relative_to(ROOT)}")
