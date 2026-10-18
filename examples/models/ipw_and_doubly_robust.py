import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from probability.distributions import Ratio, Boolean, StudentsT
from probability.models import IPW, AIPW


def main():
    print("=" * 80)
    print("PROPENSITY SCORES, IPW, AND DOUBLY ROBUST (AIPW) CAUSAL ESTIMATION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Simulating an Observational Study with Confounding
    # Scenario: Evaluating the causal impact of an automated email campaign (treatment)
    # on customer 30-day purchases (outcome).
    # Confounders:
    #   - recency: Days since last visit (smaller = more active)
    #   - historical_spend: Past customer lifetime value
    #   - mobile_user: Binary indicator for app usage
    # True Causal Effect: +$15.00 per customer.
    # -------------------------------------------------------------------------
    np.random.seed(42)
    n = 600

    recency = np.random.exponential(scale=20.0, size=n)
    historical_spend = np.random.gamma(shape=3.0, scale=30.0, size=n)
    mobile_user = (np.random.rand(n) < 0.45).astype(float)

    # Propensity score logits: active, high-spend, mobile users receive campaign more often
    z = (
        -1.0
        - 0.03 * (recency - 20.0)
        + 0.015 * (historical_spend - 90.0)
        + 0.6 * (mobile_user - 0.45)
    )
    true_propensity = 1.0 / (1.0 + np.exp(-z))
    email_campaign = (np.random.rand(n) < true_propensity).astype(float)

    # Outcome DGP: baseline + true treatment effect ($15.00) + confounders + noise
    noise = np.random.normal(0, scale=12.0, size=n)
    purchases = (
        40.0
        + 15.0 * email_campaign
        - 0.5 * recency
        + 0.4 * historical_spend
        + 8.0 * mobile_user
        + noise
    )

    df = pd.DataFrame({
        'purchases': purchases,
        'email_campaign': email_campaign,
        'recency': recency,
        'historical_spend': historical_spend,
        'mobile_user': mobile_user
    })

    print(f"Sample size: {n} customers")
    print(f"Campaign reach: {email_campaign.mean():.1%}")

    # -------------------------------------------------------------------------
    # 2. Raw Unadjusted Difference (Biased!)
    # -------------------------------------------------------------------------
    treated_mean = purchases[email_campaign == 1.0].mean()
    control_mean = purchases[email_campaign == 0.0].mean()
    naive_diff = treated_mean - control_mean
    print(f"\nRaw Unadjusted Difference: ${naive_diff:.2f}")
    print(f"(Biased upward from true effect $15.00 due to confounding by historical spend & recency)")

    # -------------------------------------------------------------------------
    # 3. Inverse Probability Weighting (IPW)
    # -------------------------------------------------------------------------
    covariates = df[['recency', 'historical_spend', 'mobile_user']]
    ipw_model = IPW.fit(
        treatment=df['email_campaign'],
        outcome=df['purchases'],
        covariates=covariates,
        estimand='ATE'
    )

    print("\n--- 1. IPW Estimation (Hajek Normalized Weights) ---")
    print(ipw_model.summary().round(3).to_string())
    print(f"IPW Estimated ATE: ${ipw_model.estimate:.2f} (SE = ${ipw_model.std_err:.2f})")

    # -------------------------------------------------------------------------
    # 4. Covariate Balance Diagnostics & Love Plot
    # -------------------------------------------------------------------------
    bal_df = ipw_model.balance_summary()
    print("\n--- Covariate Balance Table (Standardized Mean Differences) ---")
    print(bal_df.round(4).to_string())

    # -------------------------------------------------------------------------
    # 5. Augmented IPW (AIPW) / Doubly Robust Estimator
    # -------------------------------------------------------------------------
    aipw_model = AIPW.fit(
        treatment=df['email_campaign'],
        outcome=df['purchases'],
        covariates=covariates,
        estimand='ATE'
    )

    print("\n--- 2. Doubly Robust (AIPW) Estimation ---")
    print(aipw_model.summary().round(3).to_string())
    print(f"Doubly Robust Estimated ATE: ${aipw_model.estimate:.2f} (SE = ${aipw_model.std_err:.2f})")

    # Distribution-first sampling distribution
    eff_dist = aipw_model.effect_distribution
    assert isinstance(eff_dist, StudentsT)
    print("\n--- 3. Distribution-First Treatment Effect ---")
    print(f"Causal Effect Distribution: StudentsT(nu={eff_dist.nu:.0f}, mu={eff_dist.mu:.3f}, sigma={eff_dist.sigma:.3f})")
    print(f"P(Causal Effect > $10.00):  {float(eff_dist > 10.0):.2%}")
    print(f"P(Causal Effect > $0.00):   {float(eff_dist > 0.0):.4%}")
    print(f"95% Confidence/HDI Interval: ${eff_dist.interval(0.95)[0]:.2f} to ${eff_dist.interval(0.95)[1]:.2f}")

    # -------------------------------------------------------------------------
    # 6. Balance Diagnostics Plots: Love Plot & Propensity Overlap
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ipw_model.love_plot(
        threshold=0.1,
        ax=axes[0],
        title="Love Plot: Covariate Balance Before & After IPW Weighting"
    )

    ipw_model.plot_propensity(
        ax=axes[1],
        title="Propensity Score Common Support (Treated vs Control)"
    )

    plt.tight_layout()
    print("\nGenerated Love Plot and Propensity Overlap plot successfully.")


if __name__ == "__main__":
    main()
