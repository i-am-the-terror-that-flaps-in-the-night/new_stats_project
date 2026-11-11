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


def regression(df, outcome, predictors, weight, psus, level=0.95):
    # Survey-weighted linear regression of `outcome` on `predictors` (columns of
    # df), with an intercept. Teens missing any of them are left out of the fit
    # but kept in df, so their PSUs still count in the variance (domain
    # estimation, as in design_psus). Standard errors are the Taylor-linearized
    # sandwich; CIs and p-values use the design degrees of freedom.
    # Returns one row per predictor (estimate, CI, p) plus the fit's n, R², and
    # the p-value of the overall test that every predictor's effect is zero.
    cols = [outcome, *predictors]
    used = df[cols].notna().all(axis=1) & (df[weight] > 0)
    X = np.column_stack([np.ones(len(df)), df[predictors].astype(float).fillna(0)])
    y = df[outcome].astype(float).fillna(0).to_numpy()
    w = df[weight].where(used, 0).to_numpy()

    bread = np.linalg.inv(X.T @ (w[:, None] * X))
    beta = bread @ X.T @ (w * y)
    resid = y - X @ beta
    # Each teen's influence on the coefficients (zero for teens not in the fit)
    influence = (w * resid)[:, None] * X @ bread
    cov = linearized_cov(df, influence, psus)
    se = np.sqrt(np.diag(cov))

    dof = design_df(psus)
    t = stats.t.ppf(0.5 + level / 2, dof)
    p = 2 * stats.t.sf(np.abs(beta / se), dof)
    coefs = pd.DataFrame(
        {"estimate": beta, "low": beta - t * se, "high": beta + t * se, "p": p},
        index=["Intercept", *predictors],
    ).drop("Intercept")

    ybar = np.average(y, weights=w)
    r2 = 1 - np.sum(w * resid**2) / np.sum(w * (y - ybar) ** 2)

    # Overall test: adjusted Wald F on every slope at once (the intercept is
    # left out), as in compare_categories
    slopes = beta[1:]
    wald = slopes @ np.linalg.solve(cov[1:, 1:], slopes)
    q = len(slopes)
    f = wald * (dof - q + 1) / (dof * q)
    overall_p = stats.f.sf(f, q, dof - q + 1)
    return coefs, {"n": int(used.sum()), "r2": r2, "p": overall_p}


def logistic_regression(df, outcome, predictors, weight, psus, level=0.95, max_iter=50):
    # Survey-weighted logistic regression of a yes/no `outcome` (1/0) on
    # `predictors`, with an intercept. Same design handling as regression():
    # teens missing anything are left out of the fit but kept for the variance.
    # Fitted by iteratively reweighted least squares. Returns one row per
    # predictor on the log-odds scale (estimate, CI, p): exp() them for odds
    # ratios. The fit dict has n and the number of teens with the outcome.
    cols = [outcome, *predictors]
    used = df[cols].notna().all(axis=1) & (df[weight] > 0)
    X = np.column_stack([np.ones(len(df)), df[predictors].astype(float).fillna(0)])
    y = df[outcome].astype(float).fillna(0).to_numpy()
    w = df[weight].where(used, 0).to_numpy()

    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        p = 1 / (1 + np.exp(-X @ beta))
        bread = np.linalg.inv(X.T @ ((w * p * (1 - p))[:, None] * X))
        step = bread @ X.T @ (w * (y - p))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = 1 / (1 + np.exp(-X @ beta))
    bread = np.linalg.inv(X.T @ ((w * p * (1 - p))[:, None] * X))
    # Each teen's influence on the coefficients (zero for teens not in the fit)
    influence = (w * (y - p))[:, None] * X @ bread
    cov = linearized_cov(df, influence, psus)
    se = np.sqrt(np.diag(cov))

    dof = design_df(psus)
    t = stats.t.ppf(0.5 + level / 2, dof)
    pval = 2 * stats.t.sf(np.abs(beta / se), dof)
    coefs = pd.DataFrame(
        {"estimate": beta, "low": beta - t * se, "high": beta + t * se, "p": pval},
        index=["Intercept", *predictors],
    ).drop("Intercept")
    return coefs, {"n": int(used.sum()), "events": int((used & (y == 1)).sum())}
