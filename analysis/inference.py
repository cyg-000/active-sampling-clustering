"""Small, explicit inferential helpers used by revision gap tests.

Bayes factors are Jeffreys-Zellner-Siow (JZS) t-test Bayes factors with the
default Cauchy scale r=sqrt(2)/2, following Rouder et al. (2009).  The same
formula is already used by the behavioural and capacity analyses in this repo.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import integrate, stats


JZS_R = math.sqrt(2) / 2


def jzs_bf10(t: float, df: float, n_eff: float, r: float = JZS_R) -> float:
    """JZS BF10 for a t statistic; n_eff=n for one-sample/paired tests."""
    if not np.isfinite(t) or df <= 0 or n_eff <= 0:
        return np.nan

    def integrand(g):
        likelihood = (1 + n_eff * g) ** -0.5
        likelihood *= (1 + t * t / ((1 + n_eff * g) * df)) ** (-(df + 1) / 2)
        prior = r / (math.sqrt(2 * math.pi) * g ** 1.5) * math.exp(-(r * r) / (2 * g))
        return likelihood * prior

    m1, _ = integrate.quad(integrand, 1e-8, np.inf, epsabs=1e-10, epsrel=1e-8, limit=300)
    m0 = (1 + t * t / df) ** (-(df + 1) / 2)
    return float(m1 / m0)


def one_sample(values, mu=0.0) -> dict[str, float]:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    n = len(x)
    mean, sd = float(np.mean(x)), float(np.std(x, ddof=1))
    se = sd / math.sqrt(n)
    t = (mean - mu) / se
    df = n - 1
    p = float(2 * stats.t.sf(abs(t), df))
    critical = stats.t.ppf(.975, df)
    bf10 = jzs_bf10(t, df, n)
    return {"n": n, "mean_difference": mean - mu, "sd_difference": sd,
            "ci_low": mean - mu - critical * se, "ci_high": mean - mu + critical * se,
            "t": t, "df": df, "p_two_sided": p, "cohen_dz": (mean - mu) / sd,
            "bf10": bf10, "bf01": 1 / bf10}


def independent(x, y) -> dict[str, float]:
    """Equal-variance independent t/JZS BF; difference is mean(x)-mean(y)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    n1, n2 = len(x), len(y)
    result = stats.ttest_ind(x, y, equal_var=True)
    df = n1 + n2 - 2
    pooled = math.sqrt(((n1 - 1) * np.var(x, ddof=1) + (n2 - 1) * np.var(y, ddof=1)) / df)
    difference = float(np.mean(x) - np.mean(y))
    se = pooled * math.sqrt(1 / n1 + 1 / n2)
    critical = stats.t.ppf(.975, df)
    n_eff = n1 * n2 / (n1 + n2)
    bf10 = jzs_bf10(float(result.statistic), df, n_eff)
    return {"n1": n1, "n2": n2, "mean_difference": difference,
            "ci_low": difference - critical * se, "ci_high": difference + critical * se,
            "t": float(result.statistic), "df": df, "p_two_sided": float(result.pvalue),
            "cohen_d": difference / pooled, "bf10": bf10, "bf01": 1 / bf10}

