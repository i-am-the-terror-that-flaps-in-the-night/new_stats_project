import pandas as pd

from analysis.engine.tables.table_B1 import format_p
from analysis.engine.tables.table_B6 import BASE, MODELS, OUTCOME, model_variables
from analysis.paths import ROOT, TABLES_DIR
from analysis.utilities.cleaning import data_cutting
from analysis.utilities.survey import design_psus, regression, weighted_mean_sd

OUTPUT = TABLES_DIR / "table_B16.csv"

# Model B with BMI, with the insulin-resistance marker swapped: the old way
# (Trig/HDL), the new way (HOMA-IR), and both plus HbA1c
_, SAMPLE, WEIGHT, _ = MODELS[2]
HOMA_CUTOFF = 3.16  # the most-cited teen HOMA-IR cutoff (none is agreed)

VARIANTS = [
    ("With Trig/HDL ratio (the old way)", ["TG_HDL"]),
    ("With HOMA-IR (the new way)", ["LOG_HOMA_IR"]),
    ("With both, plus HbA1c", ["LOG_HOMA_IR", "TG_HDL", "LBXGH"]),
]
MARKERS = {"TG_HDL": ("Trig/HDL", "per unit"), "LOG_HOMA_IR": ("HOMA-IR", "per log unit"),
           "LBXGH": ("HbA1c", "per %")}


def p_text(p):
    p = format_p(p)
    return f"p {p if p.startswith('<') else '= ' + p}"


def table_b16(samples=None):
    if samples is None:
        samples = data_cutting()
    psus = design_psus()
    df = model_variables(samples[SAMPLE])
    # Same teens in every row: those with every marker
    every = [*BASE, "BMI_Z", "TG_HDL", "LOG_HOMA_IR", "LBXGH"]
    same = df[[OUTCOME, *every]].notna().all(axis=1) & (df[WEIGHT] > 0)
    df["Y"] = df[OUTCOME].where(same)

    rows = []
    for name, markers in VARIANTS:
        coefs, fit = regression(df, "Y", [*BASE, *markers, "BMI_Z"], WEIGHT, psus)
        # One marker: "X 7.0 per unit, p"; several: short form for each
        if len(markers) == 1:
            label, unit = MARKERS[markers[0]]
            marker = f"{label} {coefs.loc[markers[0], 'estimate']:.1f} {unit}, {p_text(coefs.loc[markers[0], 'p'])}"
        else:
            marker = "; ".join(f"{MARKERS[m][0]} {coefs.loc[m, 'estimate']:.1f}, {p_text(coefs.loc[m, 'p'])}"
                               for m in markers)
        sugar = coefs.loc["SUGAR"]
        rows.append([name, fit["n"], f"{fit['r2']:.2f}", marker, f"{sugar['estimate']:+.1f}, {p_text(sugar['p'])}"])
    table = pd.DataFrame(rows, columns=["Model B with BMI", "Teens", "R²", "Marker effect (dB/m), p",
                                        "Sugar effect (per 50 g), p"])

    # HOMA-IR in these teens: weighted averages and the share above the cutoff
    teens = df[same]
    fat = teens["LIVER_FAT"] == 1
    average = {
        "overall": weighted_mean_sd(teens["HOMA_IR"], teens[WEIGHT])[0],
        "with liver fat": weighted_mean_sd(teens.loc[fat, "HOMA_IR"], teens.loc[fat, WEIGHT])[0],
        "without": weighted_mean_sd(teens.loc[~fat, "HOMA_IR"], teens.loc[~fat, WEIGHT])[0],
    }
    above = weighted_mean_sd((teens["HOMA_IR"] > HOMA_CUTOFF).astype(float), teens[WEIGHT])[0]
    return table, average, above


if __name__ == "__main__":
    table, average, above = table_b16()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 80):
        print(table.to_string(index=False))
    print(f"\nHow to read this: Same {table['Teens'].iloc[0]} teens in every row. Average HOMA-IR: "
          + ", ".join(f"{value:.1f} {name}" for name, value in average.items())
          + f". {100 * above:.0f}% of these teens (weighted) are above the most-cited cutoff, {HOMA_CUTOFF}.")
    print(f"\nWrote {OUTPUT.relative_to(ROOT)}")
