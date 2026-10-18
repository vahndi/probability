import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from probability.distributions import Ratio, StudentsT
from probability.models import LinearRegression, OLS


def main():
    print("=" * 80)
    print("CAUSAL OLS REGRESSION & FRISCH-WAUGH-LOVELL (FWL) RESIDUALIZATION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Simulating Causal Observational Data with Confounding
    # Scenario (inspired by Causal Inference in Python):
    # Estimating the causal effect of an in-app premium coaching feature (treatment)
    # on customer revenue (outcome).
    # Confounder: User Account Size (account_size)
    # Larger accounts are more likely to adopt premium coaching AND spend more revenue.
    # True causal effect of coaching: +$25.00 per user.
    # -------------------------------------------------------------------------
    np.random.seed(42)
    n = 300

    account_size = np.random.uniform(10, 100, size=n)  # Confounder
    prob_coaching = 1.0 / (1.0 + np.exp(-(0.05 * (account_size - 50))))
    coaching = (np.random.rand(n) < prob_coaching).astype(float)  # Treatment: 0 or 1

    # True DGP: revenue = 50 + 25.0 * coaching + 1.5 * account_size + heteroskedastic noise
    noise = np.random.normal(0, scale=3.0 + 0.1 * account_size, size=n)
    revenue = 50.0 + 25.0 * coaching + 1.5 * account_size + noise

    df = pd.DataFrame({
        'revenue': revenue,
        'coaching': coaching,
        'account_size': account_size
    })

    print(f"Sample size: {n} accounts")
    print(f"Coaching adoption rate: {coaching.mean():.1%}")

    # -------------------------------------------------------------------------
    # 2. Naive Regression (Omitted Variable Bias)
    # -------------------------------------------------------------------------
    naive_model = LinearRegression.fit(
        X=df[['coaching']],
        y=df['revenue'],
        cov_type='HC3'
    )
    print("\n--- 1. Naive Model (Omitted Variable Bias: Account Size omitted) ---")
    print(naive_model.summary().round(3).to_string())
    naive_beta = naive_model.params['coaching']
    print(f"Naive estimated coaching effect: ${naive_beta:.2f} (Severely biased upward due to confounding!)")

    # -------------------------------------------------------------------------
    # 3. Controlled Causal Regression with HC3 Robust Standard Errors
    # -------------------------------------------------------------------------
    causal_model = LinearRegression.fit(
        X=df[['coaching', 'account_size']],
        y=df['revenue'],
        cov_type='HC3'
    )
    print("\n--- 2. Controlled Model (Controlling for Account Size with HC3 Robust SEs) ---")
    print(causal_model.summary().round(3).to_string())
    causal_beta = causal_model.params['coaching']
    print(f"Controlled estimated coaching effect: ${causal_beta:.2f} (True causal effect is $25.00!)")

    # -------------------------------------------------------------------------
    # 4. First-Class Student's t Sampling Distribution
    # -------------------------------------------------------------------------
    beta_dist = causal_model.coefficient('coaching')
    assert isinstance(beta_dist, StudentsT)
    print("\n--- 3. Distribution-First Treatment Effect ---")
    print(f"Treatment Distribution: StudentsT(nu={beta_dist.nu:.0f}, mu={beta_dist.mu:.3f}, sigma={beta_dist.sigma:.3f})")
    print(f"P(Causal Effect > $20.00): {float(beta_dist > 20.0):.2%}")
    print(f"P(Causal Effect > $0.00):  {float(beta_dist > 0.0):.4%}")
    print(f"95% HDI Credible/Confidence Interval: {beta_dist.interval(0.95)}")

    # -------------------------------------------------------------------------
    # 5. Ratio.residualize: Creating the Confounder-Adjusted Outcome
    # -------------------------------------------------------------------------
    raw_revenue_ratio = Ratio(df['revenue'])
    adjusted_revenue_ratio = raw_revenue_ratio.residualize(
        df[['account_size']],
        keep_mean=True
    )
    print("\n--- 4. Ratio Confounder Residualization (FWL Partialling-Out) ---")
    print(f"Raw Revenue SD:      ${raw_revenue_ratio.std():.2f}")
    print(f"Adjusted Revenue SD: ${adjusted_revenue_ratio.std():.2f} (Confounder variance removed!)")
    print(f"Adjusted Mean:       ${adjusted_revenue_ratio.mean():.2f} (Mean preserved on original scale)")

    # -------------------------------------------------------------------------
    # 6. Frisch-Waugh-Lovell Decomposition Verification
    # -------------------------------------------------------------------------
    tilde_D, tilde_y, fwl_bivariate = causal_model.fwl_decompose('coaching')
    fwl_beta = float(fwl_bivariate.params['coaching_residualized'])
    print("\n--- 5. Frisch-Waugh-Lovell Theorem Verification ---")
    print(f"Multivariate OLS beta: ${causal_beta:.6f}")
    print(f"FWL Bivariate OLS beta: ${fwl_beta:.6f}")
    print(f"Difference:             ${abs(causal_beta - fwl_beta):.1e} (Analytically identical!)")

    # -------------------------------------------------------------------------
    # 7. Visualization: Coefficient Forest Plot & FWL Partialled Scatter Plot
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    causal_model.plot_coefficients(
        hdi=0.95,
        ax=axes[0],
        title="Causal Regression Coefficients (95% CI, HC3 SE)"
    )

    causal_model.plot_fwl(
        treatment_var='coaching',
        hdi=0.95,
        ax=axes[1],
        title="FWL Partialled-Out Causal Effect: Coaching on Revenue"
    )

    plt.tight_layout()
    print("\nGenerated plots successfully.")


if __name__ == "__main__":
    main()
