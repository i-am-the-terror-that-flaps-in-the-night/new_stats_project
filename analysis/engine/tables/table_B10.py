import pandas as pd

from analysis.engine.figures.figure_B7 import sugar_quarters
from analysis.engine.tables.table_B1 import format_p
from analysis.paths import ROOT, TABLES_DIR

OUTPUT = TABLES_DIR / "table_B10.csv"


def table_b10(samples=None):
    # Figure B7's sugar quarters as a table, plus the trend-test p-values for
    # its "how to read this" note
    quarters, trend = sugar_quarters(samples)
    rows = [[
        quarter,
        q["range"],
        int(q["n"]),
        f"{q['sugar']:.0f}",
        f"{q['cap']:.0f} ({q['cap_low']:.0f}–{q['cap_high']:.0f})",
        f"{q['fat']:.0f}% ({q['fat_low']:.0f}–{q['fat_high']:.0f})",
    ] for quarter, q in quarters.iterrows()]
    table = pd.DataFrame(rows, columns=[
        "Quarter", "Sugar range (g/day)", "Teens", "Average sugar",
        "Average CAP (95% CI)", "% with liver fat (95% CI)",
    ])
    return table, trend


if __name__ == "__main__":
    table, trend = table_b10()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)
    print(table.to_string(index=False))
    (cap, cap_adj), (fat, fat_adj) = trend["LUXCAPM"], trend["LIVER_FAT"]
    print(f"\nHow to read this: Trend p-values: CAP {format_p(cap)}, liver fat present {format_p(fat)}; "
          f"adjusted for the Model A factors, {format_p(cap_adj)} and {format_p(fat_adj)}.")
    print(f"\nWrote {OUTPUT.relative_to(ROOT)}")
