import pandas as pd

from analysis.engine.figures.figure_B11 import visceral_thirds
from analysis.engine.tables.table_B1 import format_p
from analysis.paths import ROOT, TABLES_DIR

OUTPUT = TABLES_DIR / "table_visceral_thirds.csv"


def table_visceral_thirds(samples=None):
    # The unnumbered table in section B11 above Figure B11: the figure's
    # visceral-fat thirds within each weight group, as numbers
    thirds, trend = visceral_thirds(samples)
    rows = []
    for (group, third), t in thirds.iterrows():
        rows.append([
            group if third == "Lowest third" else "",
            third.replace(" third", ""),
            int(t["n"]),
            f"{t['visceral']:.0f}",
            f"{t['cap']:.0f} ({t['cap_low']:.0f}–{t['cap_high']:.0f})",
            f"{t['fat']:.0f}% ({t['fat_low']:.0f}–{t['fat_high']:.0f})",
        ])
    table = pd.DataFrame(rows, columns=["Weight group", "Visceral-fat third", "Teens", "Average visceral fat (g)",
                                        "Average CAP (95% CI)", "% with liver fat (95% CI)"])
    return table, trend


if __name__ == "__main__":
    table, trend = table_visceral_thirds()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)
    print(table.to_string(index=False))
    print("\nHow to read this: Trend across thirds: "
          + "; ".join(f"{group} p {format_p(p) if format_p(p).startswith('<') else '= ' + format_p(p)}"
                      for group, p in trend.items()) + ".")
    print(f"\nWrote {OUTPUT.relative_to(ROOT)}")
