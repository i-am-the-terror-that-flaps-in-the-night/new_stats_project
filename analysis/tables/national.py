import pandas as pd

from analysis.paths import DATA_DIR, ROOT
from analysis.tables.cohort import RACE, add_variables
from analysis.utilities.cleaning import clean, data_cutting
from analysis.utilities.survey import design_psus, mean_ci

OUTPUT = ROOT / "output" / "table_B2.csv"

WEIGHT = "WTMEC2YR"  # exam weight: the screening sample comes from the exam

# Underweight is folded into healthy weight here (too few teens to stand alone)
WEIGHT_GROUP_B2 = {
    "Underweight": "Healthy weight (incl. underweight)",
    "Healthy weight": "Healthy weight (incl. underweight)",
    "Overweight": "Overweight",
    "Obesity": "Obesity",
}


def groups(df):
    # Each row of the table: label and who is in it
    yield "All adolescents 12–17", pd.Series(True, index=df.index)
    yield "Males", df["BOY"]
    yield "Females", ~df["BOY"]
    weight_group = df["WEIGHT_GROUP"].astype(object).map(WEIGHT_GROUP_B2)
    for label in dict.fromkeys(WEIGHT_GROUP_B2.values()):
        yield label, weight_group == label
    for label in dict.fromkeys(RACE.values()):
        yield label, df["RACE"] == label


def examined_teens(age_min=12, age_max=17):
    # Every teen examined at the mobile clinic. Their exam weights add up to the
    # U.S. teen population. The scanned teens' weights alone add up to less
    # (about 19 vs. 25 million), because the weights aren't adjusted for teens
    # who missed the scan, so group sizes come from here.
    df = clean(pd.read_parquet(DATA_DIR / "merged" / "nhanes_J.parquet"))
    df = df[(df["RIDSTATR"] == 2) & df["RIDAGEYR"].between(age_min, age_max)]
    return add_variables(df)


def pct(estimate):
    # Percentage with its CI, clipped at 0 (a share can't be negative)
    mean, low, high = (100 * x for x in estimate)
    return f"{mean:.1f}% ({max(low, 0):.1f}–{high:.1f})"


def table_b2(samples=None):
    if samples is None:
        samples = data_cutting()
    psus = design_psus()
    df = add_variables(samples["screening"])
    everyone = examined_teens()
    us_teens = {label: everyone.loc[in_group, WEIGHT].sum() / 1e6
                for label, in_group in groups(everyone)}

    rows = []
    for label, in_group in groups(df):
        # U.S. teens in the group (millions): exam weights of all examined teens
        population = us_teens[label]
        fat = mean_ci(df, "LIVER_FAT", WEIGHT, in_group, psus)
        stiff = mean_ci(df, "HIGH_STIFFNESS", WEIGHT, in_group, psus)
        # Teens with liver fat = share with liver fat x teens in the group
        mean, low, high = (population * x for x in fat)
        rows.append([
            label,
            in_group.sum(),
            f"{population:.1f}",
            pct(fat),
            f"{mean:.1f} ({max(low, 0):.1f}–{high:.1f})",
            pct(stiff),
        ])

    return pd.DataFrame(rows, columns=[
        "Group",
        "Teens scanned, n",
        "U.S. teens, millions",
        "With liver fat, weighted % (95% CI)",
        "U.S. teens with liver fat, millions (95% CI)",
        "High stiffness, weighted % (95% CI)",
    ])


if __name__ == "__main__":
    table = table_b2()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)
    with pd.option_context("display.width", 250):
        print(table.to_string(index=False))
    print(f"\nWrote {OUTPUT.relative_to(ROOT)}")
