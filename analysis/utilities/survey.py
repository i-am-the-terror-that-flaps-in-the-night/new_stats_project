import numpy as np
import pandas as pd
from scipy import stats

from analysis.paths import DATA_DIR

STRATA = "SDMVSTRA"
PSU = "SDMVPSU"


def design_psus():
    # Every (stratum, PSU) pair in the full NHANES file. Variances are summed over
    # all of them, so a PSU with nobody in a sample still counts (as zero). This is
    # domain estimation: the same as keeping everyone in the file and marking who
    # is in each sample, which subsetting first would get wrong.
    df = pd.read_parquet(DATA_DIR / "merged" / "nhanes_J.parquet", columns=[STRATA, PSU])
    return pd.MultiIndex.from_frame(df.drop_duplicates().sort_values([STRATA, PSU]))


def design_df(psus):
    # NHANES degrees of freedom: number of PSUs minus number of strata
    return len(psus) - psus.get_level_values(0).nunique()


def linearized_cov(df, influence, psus):
    # Taylor-linearization covariance from per-person influence values (one column
    # per estimate): total them by PSU, then compare PSUs within each stratum.
    # Uses the with-replacement formula n_h / (n_h - 1) * sum (Z_hi - mean_h)^2.
    z = pd.DataFrame(influence, index=df.index)
    totals = z.groupby([df[STRATA], df[PSU]]).sum().reindex(psus, fill_value=0)
    centered = totals - totals.groupby(level=0).transform("mean")
    n_h = pd.Series(totals.index.get_level_values(0), index=totals.index).map(
        totals.groupby(level=0).size()
    )
    scaled = centered.mul(np.sqrt(n_h / (n_h - 1)), axis=0).to_numpy()
    return scaled.T @ scaled


def weighted_mean_sd(y, w):
    # Weighted mean and population standard deviation, skipping missing values
    ok = y.notna() & w.notna()
    y, w = y[ok], w[ok]
    mean = np.average(y, weights=w)
    return mean, np.sqrt(np.average((y - mean) ** 2, weights=w))


def _group_mean_influence(y, w, in_group):
    # Influence values for a weighted mean within a group (zero outside it)
    w = w.where(in_group & y.notna(), 0)
    mean = (w * y.fillna(0)).sum() / w.sum()
    return mean, (w * (y.fillna(0) - mean) / w.sum()).to_numpy()


def compare_means(df, col, weight, group, psus):
    # Design-based t-test that a variable's weighted mean is the same in both
    # levels of `group` (a boolean Series). Returns the two-sided p-value.
    y = df[col].astype(float)
    m1, z1 = _group_mean_influence(y, df[weight], group)
    m0, z0 = _group_mean_influence(y, df[weight], ~group)
    se = np.sqrt(linearized_cov(df, (z1 - z0)[:, None], psus)[0, 0])
    t = (m1 - m0) / se
    return 2 * stats.t.sf(abs(t), design_df(psus))


def compare_categories(df, col, weight, group, psus):
    # Design-based test that a categorical variable has the same distribution in
    # both levels of `group`: adjusted Wald F on the differences in weighted
    # proportions (the last category is left out, since they sum to 1).
    levels = df[col].dropna().unique()
    if len(levels) < 2:
        return np.nan
    diffs, influence = [], []
    for level in sorted(levels)[:-1]:
        y = (df[col] == level).astype(float).where(df[col].notna())
        m1, z1 = _group_mean_influence(y, df[weight], group)
        m0, z0 = _group_mean_influence(y, df[weight], ~group)
        diffs.append(m1 - m0)
        influence.append(z1 - z0)

    d = np.array(diffs)
    cov = linearized_cov(df, np.column_stack(influence), psus)
    wald = d @ np.linalg.solve(cov, d)
    q, dof = len(d), design_df(psus)
    f = wald * (dof - q + 1) / (dof * q)
    return stats.f.sf(f, q, dof - q + 1)


def mean_ci(df, col, weight, in_group, psus, level=0.95):
    # Weighted mean of `col` within a group (a boolean Series), with a design-based
    # confidence interval: mean +/- t * SE, using the design degrees of freedom
    y = df[col].astype(float)
    mean, z = _group_mean_influence(y, df[weight], in_group)
    se = np.sqrt(linearized_cov(df, z[:, None], psus)[0, 0])
    t = stats.t.ppf(0.5 + level / 2, design_df(psus))
    return mean, mean - t * se, mean + t * se
