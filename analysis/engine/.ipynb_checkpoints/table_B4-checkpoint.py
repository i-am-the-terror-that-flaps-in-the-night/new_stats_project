import pandas as pd

from analysis.engine.table_B1 import add_variables
from analysis.engine.table_B2 import WEIGHT, examined_teens
from analysis.paths import ROOT
from analysis.utilities.cleaning import data_cutting
from analysis.utilities.survey import design_psus, mean_ci

OUTPUT = ROOT / "output" / "table_B4.csv"

# Waist-to-height ratio at or above this counts as belly fat
WHTR_HIGH = 0.5


def groups(df):
    # Each row of the table: label and who is in it. Underweight is folded into
    # healthy weight, as in Table B2 and Figure B3.
    healthy = df["WEIGHT_GROUP"].isin(["Underweight", "Healthy weight"])
    obese = df["WEIGHT_GROUP"] == "Obesity"
    normal_alt = df["HIGH_ALT"] == 0
    yield "Healthy weight", healthy
    yield "Overweight", df["WEIGHT_GROUP"] == "Overweight"
    yield "Obesity", obese
    yield "Normal ALT (pediatric cutoffs)", normal_alt
    yield "Not obese AND normal ALT", ~obese & normal_alt
    yield (f"Healthy weight AND normal ALT AND waist-to-height <{WHTR_HIGH}",
           healthy & normal_alt & (df["WHTR"] < WHTR_HIGH))


def table_b4(samples=None):
    if samples is None:
        samples = data_cutting()
    psus = design_psus()
    df = add_variables(samples["screening"])
    has_fat = df["LIVER_FAT"] == 1

    # U.S. teens with liver fat (millions), worked out as in Table B2: the share
    # with liver fat times all examined teens
    population = examined_teens()[WEIGHT].sum() / 1e6
    share_fat, _, _ = mean_ci(df, "LIVER_FAT", WEIGHT, pd.Series(True, index=df.index), psus)
    with_fat = population * share_fat

    rows = []
    for label, in_group in groups(df):
        # Share of teens with liver fat who are in the group: the weighted mean
        # of a yes/no column, taken among teens with liver fat only
        df["IN_GROUP"] = in_group.astype(float)
        mean, low, high = (100 * x for x in mean_ci(df, "IN_GROUP", WEIGHT, has_fat, psus))
        rows.append([
            label,
            (has_fat & in_group).sum(),
            f"{mean:.0f}% ({max(low, 0):.0f}–{high:.0f})",
            f"{with_fat * mean / 100:.1f}",
        ])

    return pd.DataFrame(rows, columns=[
        f"Among teens WITH liver fat (n = {has_fat.sum()}), how many are…",
        "Number",
        "Weighted % (95% CI)",
        "U.S. teens, millions",
    ])


if __name__ == "__main__":
    table = table_b4()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)
    with pd.option_context("display.width", 250):
        print(table.to_string(index=False))
    print(f"\nWrote {OUTPUT.relative_to(ROOT)}")
