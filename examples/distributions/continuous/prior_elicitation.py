"""
Demonstration of Prior Elicitation from Quantiles and Intervals (Phase 6.1).

In applied Bayesian data analysis, domain experts rarely think in terms of
abstract shape parameters (such as alpha and beta in a Beta or Gamma distribution).
Instead, experts naturally express knowledge in terms of credible intervals,
such as:
  - "I am 95% confident the conversion rate is between 4% and 12%."
  - "The average customer wait time is between 2 and 10 minutes (90% interval)."
  - "The asset price at year end is likely between $50 and $200 (95% interval)."

This script demonstrates solving for exact distribution parameters from intervals
and arbitrary quantiles across Beta, Gamma, LogNormal, and Normal distributions.
"""

from probability.distributions import (
    Beta,
    Gamma,
    LogNormal,
    Normal,
)


def elicit_conversion_rate_prior():
    print("=== 1. Beta Distribution: Conversion Rate Prior Elicitation ===")
    # Expert belief: 95% credible interval for click-through rate is [0.04, 0.12]
    prior_beta = Beta.from_interval(lower=0.04, upper=0.12, prob=0.95)
    print(f"Elicited: {prior_beta!r}")
    print(f"Alpha: {prior_beta.alpha:.3f}, Beta: {prior_beta.beta:.3f}")
    lower, upper = prior_beta.interval(0.95)
    print(f"Verified 95% interval: [{lower:.4f}, {upper:.4f}]")
    print(f"Prior mean: {prior_beta.mean():.4f}\n")


def elicit_wait_time_prior():
    print("=== 2. Gamma Distribution: Service Wait Time Prior ===")
    # Expert belief: 90% credible interval for waiting time is [2.0, 10.0] minutes
    prior_gamma = Gamma.from_interval(lower=2.0, upper=10.0, prob=0.90)
    print(f"Elicited: {prior_gamma!r}")
    print(f"Shape (alpha): {prior_gamma.alpha:.3f}, Rate (beta): {prior_gamma.beta:.3f}")
    lower, upper = prior_gamma.interval(0.90)
    print(f"Verified 90% interval: [{lower:.2f}, {upper:.2f}]")
    print(f"Prior mean: {prior_gamma.mean():.2f} minutes\n")


def elicit_financial_price_prior():
    print("=== 3. LogNormal Distribution: Asset Valuation Prior ===")
    # Expert belief: 95% credible interval for asset value is [$50, $200]
    prior_ln = LogNormal.from_interval(lower=50.0, upper=200.0, prob=0.95)
    print(f"Elicited: {prior_ln!r}")
    print(f"Log-scale mu: {prior_ln.mu:.3f}, sigma: {prior_ln.sigma:.3f}")
    lower, upper = prior_ln.interval(0.95)
    print(f"Verified 95% interval: [${lower:.1f}, ${upper:.1f}]")
    print(f"Median: ${prior_ln.median():.1f}, Mode: ${prior_ln.mode():.1f}\n")


def elicit_treatment_effect_prior():
    print("=== 4. Normal Distribution: Treatment Effect Prior ===")
    # Expert belief: 80% interval for effect size is between -0.5 and 2.5
    prior_norm = Normal.from_interval(lower=-0.5, upper=2.5, prob=0.80)
    print(f"Elicited: {prior_norm!r}")
    print(f"Mean (mu): {prior_norm.mu:.3f}, Std (sigma): {prior_norm.sigma:.3f}")
    lower, upper = prior_norm.interval(0.80)
    print(f"Verified 80% interval: [{lower:.3f}, {upper:.3f}]\n")


def elicit_from_asymmetric_quantiles():
    print("=== 5. Asymmetric Quantiles (e.g., Median and 90th Percentile) ===")
    # Suppose median (50th percentile) is 15.0 and 90th percentile is 35.0
    gamma_asym = Gamma.from_quantiles(q1=0.50, v1=15.0, q2=0.90, v2=35.0)
    print(f"Elicited Gamma: {gamma_asym!r}")
    print(f"CDF at 15.0: {gamma_asym.cdf().at(15.0):.3f} (target 0.500)")
    print(f"CDF at 35.0: {gamma_asym.cdf().at(35.0):.3f} (target 0.900)")


if __name__ == '__main__':
    elicit_conversion_rate_prior()
    elicit_wait_time_prior()
    elicit_financial_price_prior()
    elicit_treatment_effect_prior()
    elicit_from_asymmetric_quantiles()
