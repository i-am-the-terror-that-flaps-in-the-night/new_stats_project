import pandas as pd

from analysis.engine.figures.figure_B10 import SCORE_PARTS, risk_score
from analysis.paths import ROOT, TABLES_DIR

OUTPUT = TABLES_DIR / "table_B13.csv"

# Units for the medians in the "how to read this" note
UNITS = {"DR1TSUGR": " g", "SCREEN_TIME": " hrs", "TG_HDL": "", "LBXGH": "%", "BMI_Z": ""}
DECIMALS = {"DR1TSUGR": 0, "SCREEN_TIME": 0, "TG_HDL": 2, "LBXGH": 1, "BMI_Z": 2}
AUC_LABELS = {"SCORE": "score", "LBXSATSI": "ALT", "BMI_Z": "BMI z", "WHTR": "waist-to-height"}


def table_b13(samples=None):
    # Figure B10's risk score as a table, plus the medians, the rise per point
    # and the AUCs for its "how to read this" note
    scores, slope, aucs, _, medians = risk_score(samples)
    rows = [[
        score,
        int(s["n"]),
        f"{s['cap']:.0f} ({s['cap_low']:.0f}–{s['cap_high']:.0f})",
        f"{s['fat']:.0f}% ({s['fat_low']:.0f}–{min(s['fat_high'], 100):.0f})",
    ] for score, s in scores.iterrows()]
    table = pd.DataFrame(rows, columns=["Score", "Teens", "Average CAP (95% CI)", "% with liver fat (95% CI)"])
    return table, slope, aucs, medians


if __name__ == "__main__":
    table, slope, aucs, medians = table_b13()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)
    print(table.to_string(index=False))
    parts = ", ".join(f"{SCORE_PARTS[col]} (>{m:.{DECIMALS[col]}f}{UNITS[col]})" for col, m in medians.items())
    print(f"\nHow to read this: One point each for above-median {parts}, plus one for being a boy. "
          f"Liver fat rose {slope['estimate']:.0f} dB/m per point. AUC: "
          + ", ".join(f"{AUC_LABELS[col]} {auc:.2f}" for col, auc in aucs.items()) + ".")
    print(f"\nWrote {OUTPUT.relative_to(ROOT)}")
