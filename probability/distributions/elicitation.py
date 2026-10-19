from typing import Tuple

import numpy as np
from scipy.optimize import brentq, root, least_squares
from scipy.special import betainc
from scipy.stats import norm as norm_dist, gamma as gamma_dist


def elicit_normal(
        q1: float, v1: float,
        q2: float, v2: float
) -> Tuple[float, float]:
    """
    Find Normal(mu, sigma) parameters matching two quantile specifications:
        P(X <= v1) = q1 and P(X <= v2) = q2

    :param q1: Lower cumulative probability (0 < q1 < q2 < 1).
    :param v1: Quantile value at q1.
    :param q2: Upper cumulative probability (0 < q1 < q2 < 1).
    :param v2: Quantile value at q2 (v2 > v1).
    :return: (mu, sigma)
    """
    if not (0.0 < q1 < q2 < 1.0):
        raise ValueError(f"Probabilities must satisfy 0 < q1 < q2 < 1, got q1={q1}, q2={q2}.")
    if v1 >= v2:
        raise ValueError(f"Quantile values must satisfy v1 < v2, got v1={v1}, v2={v2}.")

    z1 = float(norm_dist.ppf(q1))
    z2 = float(norm_dist.ppf(q2))
    sigma = float((v2 - v1) / (z2 - z1))
    mu = float(v1 - z1 * sigma)
    return mu, sigma


def elicit_log_normal(
        q1: float, v1: float,
        q2: float, v2: float
) -> Tuple[float, float]:
    """
    Find LogNormal(mu, sigma) parameters matching two quantile specifications:
        P(X <= v1) = q1 and P(X <= v2) = q2
    where mu and sigma are the mean and standard deviation of ln(X).

    :param q1: Lower cumulative probability (0 < q1 < q2 < 1).
    :param v1: Quantile value at q1 (v1 > 0).
    :param q2: Upper cumulative probability (0 < q1 < q2 < 1).
    :param v2: Quantile value at q2 (v2 > v1).
    :return: (mu, sigma)
    """
    if v1 <= 0.0 or v2 <= 0.0:
        raise ValueError(f"LogNormal quantile values must be strictly positive, got v1={v1}, v2={v2}.")
    return elicit_normal(
        q1=q1, v1=float(np.log(v1)),
        q2=q2, v2=float(np.log(v2))
    )


def elicit_gamma(
        q1: float, v1: float,
        q2: float, v2: float
) -> Tuple[float, float]:
    """
    Find Gamma(alpha, beta) parameters matching two quantile specifications:
        P(X <= v1) = q1 and P(X <= v2) = q2
    where alpha is shape and beta is rate (mean = alpha / beta).

    :param q1: Lower cumulative probability (0 < q1 < q2 < 1).
    :param v1: Quantile value at q1 (v1 > 0).
    :param q2: Upper cumulative probability (0 < q1 < q2 < 1).
    :param v2: Quantile value at q2 (v2 > v1).
    :return: (alpha, beta)
    """
    if v1 <= 0.0 or v2 <= 0.0:
        raise ValueError(f"Gamma quantile values must be strictly positive, got v1={v1}, v2={v2}.")
    if v1 >= v2:
        raise ValueError(f"Quantile values must satisfy v1 < v2, got v1={v1}, v2={v2}.")
    if not (0.0 < q1 < q2 < 1.0):
        raise ValueError(f"Probabilities must satisfy 0 < q1 < q2 < 1, got q1={q1}, q2={q2}.")

    target_ratio = float(v2 / v1)

    # Ratio of standard gamma quantiles is strictly decreasing in alpha
    def ratio_diff(log_alpha):
        a = float(np.exp(log_alpha))
        q1_std = float(gamma_dist.ppf(q1, a=a, scale=1.0))
        q2_std = float(gamma_dist.ppf(q2, a=a, scale=1.0))
        if q1_std <= 0.0:
            return 1e12
        return float((q2_std / q1_std) - target_ratio)

    # Automatically expand bracket if necessary
    low_log_a, high_log_a = -5.0, 5.0
    f_low = ratio_diff(low_log_a)
    f_high = ratio_diff(high_log_a)

    while f_low <= 0.0 and low_log_a > -20.0:
        low_log_a -= 2.0
        f_low = ratio_diff(low_log_a)
    while f_high >= 0.0 and high_log_a < 20.0:
        high_log_a += 2.0
        f_high = ratio_diff(high_log_a)

    sol_log_a = brentq(ratio_diff, a=low_log_a, b=high_log_a, xtol=1e-10)
    alpha = float(np.exp(sol_log_a))

    # Rate beta = G_alpha(q1) / v1
    q1_val = float(gamma_dist.ppf(q1, a=alpha, scale=1.0))
    beta = float(q1_val / v1)
    return alpha, beta


def elicit_beta(
        q1: float, v1: float,
        q2: float, v2: float
) -> Tuple[float, float]:
    """
    Find Beta(alpha, beta) parameters matching two quantile specifications:
        P(X <= v1) = q1 and P(X <= v2) = q2

    :param q1: Lower cumulative probability (0 < q1 < q2 < 1).
    :param v1: Quantile value at q1 (0 < v1 < 1).
    :param q2: Upper cumulative probability (0 < q1 < q2 < 1).
    :param v2: Quantile value at q2 (0 < v1 < v2 < 1).
    :return: (alpha, beta)
    """
    if not (0.0 < v1 < v2 < 1.0):
        raise ValueError(f"Beta quantile values must satisfy 0 < v1 < v2 < 1, got v1={v1}, v2={v2}.")
    if not (0.0 < q1 < q2 < 1.0):
        raise ValueError(f"Probabilities must satisfy 0 < q1 < q2 < 1, got q1={q1}, q2={q2}.")

    # Method of moments initial guess based on normal approximation
    z1 = float(norm_dist.ppf(q1))
    z2 = float(norm_dist.ppf(q2))
    mu_approx = (v1 + v2) / 2.0
    sd_approx = max((v2 - v1) / max(z2 - z1, 1e-4), 1e-4)

    max_var = mu_approx * (1.0 - mu_approx) * 0.90
    var_approx = min(sd_approx ** 2, max_var)
    nu_approx = max(mu_approx * (1.0 - mu_approx) / max(var_approx, 1e-6) - 1.0, 0.1)
    a0 = max(mu_approx * nu_approx, 0.1)
    b0 = max((1.0 - mu_approx) * nu_approx, 0.1)

    def obj(theta):
        log_a, log_b = theta
        a = float(np.exp(log_a))
        b = float(np.exp(log_b))
        f1 = float(betainc(a, b, v1) - q1)
        f2 = float(betainc(a, b, v2) - q2)
        return [f1, f2]

    res = root(obj, x0=[np.log(a0), np.log(b0)], method='hybr')
    if not res.success or np.any(np.isnan(res.x)):
        ls_res = least_squares(obj, x0=[np.log(a0), np.log(b0)])
        opt_x = ls_res.x
    else:
        opt_x = res.x

    alpha = float(np.exp(opt_x[0]))
    beta = float(np.exp(opt_x[1]))
    return alpha, beta


def elicit_inverse_gamma(
        q1: float, v1: float,
        q2: float, v2: float
) -> Tuple[float, float]:
    """
    Find InverseGamma(alpha, beta) parameters matching two quantile specifications:
        P(Y <= v1) = q1 and P(Y <= v2) = q2
    where alpha is shape and beta is scale (mean = beta / (alpha - 1)).

    Uses the exact duality: if Y ~ InvGamma(alpha, beta), then 1/Y ~ Gamma(alpha, rate=beta).
    """
    if v1 <= 0.0 or v2 <= 0.0:
        raise ValueError(f"InverseGamma quantile values must be strictly positive, got v1={v1}, v2={v2}.")
    return elicit_gamma(
        q1=1.0 - q2, v1=1.0 / v2,
        q2=1.0 - q1, v2=1.0 / v1
    )


def elicit_continuous_uniform(
        q1: float, v1: float,
        q2: float, v2: float
) -> Tuple[float, float]:
    """
    Find ContinuousUniform(lower, upper) parameters matching two quantile specifications:
        P(X <= v1) = q1 and P(X <= v2) = q2
    """
    if not (0.0 < q1 < q2 < 1.0):
        raise ValueError(f"Probabilities must satisfy 0 < q1 < q2 < 1, got q1={q1}, q2={q2}.")
    if v1 >= v2:
        raise ValueError(f"Quantile values must satisfy v1 < v2, got v1={v1}, v2={v2}.")

    scale = (v2 - v1) / (q2 - q1)
    lower = float(v1 - q1 * scale)
    upper = float(lower + scale)
    return lower, upper
