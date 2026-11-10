import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.engine.tables.table_B1 import add_variables, format_p
from analysis.engine.tables.table_B2 import WEIGHT
from analysis.paths import FIGURES_DIR, ROOT
from analysis.utilities.cleaning import data_cutting
from analysis.utilities.survey import compare_means, design_psus

OUTPUT = FIGURES_DIR / "figure_A8.svg"

# Rows, top to bottom: label, column, decimals, and which measurement it comes
# from ("exam" = body measurements, DXA and blood tests; "fasting"; "diet").
# BMI z-score is shown as text only: a % difference of z-scores means nothing.
ROWS = [
    ("BMI z-score", "BMI_Z", 2, "exam"),
    ("Waist-to-height ratio", "WHTR", 3, "exam"),
    ("Visceral fat, g", "DXXVFATM", 0, "exam"),
    ("Subcutaneous abdominal fat, g", "DXXSATM", 0, "exam"),
    ("Android (belly) fat, %", "DXXAPFAT", 1, "exam"),
    ("Android/gynoid ratio", "DXXAGRAT", 2, "exam"),
    ("HOMA-IR", "HOMA_IR", 2, "fasting"),
    ("Fasting insulin, µU/mL", "LBXIN", 1, "fasting"),
    ("Trig/HDL ratio", "TG_HDL", 2, "fasting"),
    ("HbA1c, %", "LBXGH", 2, "fasting"),
    ("Uric acid, mg/dL", "LBXSUA", 2, "exam"),
    ("ALT, U/L", "LBXSATSI", 1, "exam"),
    ("Total sugar, g/day", "DR1TSUGR", 0, "diet"),
]
TEXT_ONLY = {"BMI_Z"}

BLUE, RED = "#2e75b6", "#c0392b"


def measured(df):
    # Who has each kind of measurement, and the weight that goes with it. As in
    # the fasting sample, a fasting weight of 0 means the teen didn't meet
    # NHANES's fasting rules, so they're left out.
    fasting = df["LBXIN"].notna() & df["LBXGLU"].notna() & (df["WTSAF2YR"] > 0)
    diet = (df["DR1DRSTZ"] == 1) & df["DR1TSUGR"].notna() & (df["WTDRD1"] > 0)
    return {
        "exam": (pd.Series(True, index=df.index), WEIGHT),
        "fasting": (fasting, "WTSAF2YR"),
        "diet": (diet, "WTDRD1"),
    }


def nonobese_comparison(samples=None):
    # Teens without obesity who have a valid DXA scan, split by liver fat. For
    # each row: weighted mean with and without liver fat, teens in each, the %
    # difference, and a design-based p-value
    if samples is None:
        samples = data_cutting()
    psus = design_psus()
    df = add_variables(samples["screening"])
    domain = (df["WEIGHT_GROUP"] != "Obesity") & (df["DXXAGST"] == 1) & df["DXXVFATM"].notna()
    has_fat = df["LIVER_FAT"] == 1
    kinds = measured(df)

    rows = []
    for label, col, decimals, kind in ROWS:
        has_kind, weight = kinds[kind]
        # The measure outside this row's teens is set to missing, so only they
        # count, but every teen's PSU stays in the variance (domain estimation)
        in_row = domain & has_kind & df[col].notna()
        df["Y"] = df[col].where(in_row)
        means = {}
        for name, part in [("with", in_row & has_fat), ("without", in_row & ~has_fat)]:
            means[name] = np.average(df.loc[part, col], weights=df.loc[part, weight])
            means[f"n_{name}"] = part.sum()
        rows.append({
            "label": label, "decimals": decimals, **means,
            "diff_pct": 100 * (means["with"] / means["without"] - 1),
            "p": compare_means(df, "Y", weight, has_fat, psus),
        })
    return pd.DataFrame(rows).set_index("label"), (domain & has_fat).sum(), (domain & ~has_fat).sum()


def figure_a8(samples=None):
    rows, n_with, n_without = nonobese_comparison(samples)

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
    y = range(len(rows))[::-1]
    for row, (label, r), (_, col, _, _) in zip(y, rows.iterrows(), ROWS):
        colour = RED if r["p"] < 0.05 else BLUE
        p = format_p(r["p"])
        d = int(r["decimals"])
        text = f"{r['with']:.{d}f} vs. {r['without']:.{d}f}"
        if col in TEXT_ONLY:
            ax.text(1, row, f"{text} (z-score)   p {p if p.startswith('<') else '= ' + p}",
                    va="center", fontsize=7.5, color=colour)
            continue
        ax.barh(row, r["diff_pct"], height=0.6, color=colour)
        ax.text(max(r["diff_pct"], 0) + 1, row, f"{text}   p {p if p.startswith('<') else '= ' + p}",
                va="center", fontsize=7.5)

    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_yticks(list(y), rows.index, fontsize=8)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(min(-10, rows["diff_pct"].min() - 5), 110)
    ax.set_xlabel("% higher in non-obese teens WITH liver fat than in those without", fontsize=9)
    ax.set_title(f"Non-obese teens: {n_with} with liver fat vs. {n_without} without (red = p < 0.05)",
                 fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return fig, rows


if __name__ == "__main__":
    fig, rows = figure_a8()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    with pd.option_context("display.width", 200):
        print(rows.drop(columns="decimals").round(3).to_string())
    print(f"\nWrote {OUTPUT.relative_to(ROOT)}")
