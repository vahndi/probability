import matplotlib.pyplot as plt
import numpy as np

from probability.models import Quap


def main():
    print("=" * 70)
    print("QUADRATIC / LAPLACE APPROXIMATION (QUAP) FOR BAYESIAN POSTERIORS")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Height vs. Centered Weight (Statistical Rethinking Ch 4)
    # Model:
    #   h_i ~ Normal(mu_i, sigma^2)
    #   mu_i = alpha + beta * (w_i - w_bar)
    # Priors:
    #   alpha ~ Normal(175, 20^2)
    #   beta  ~ Normal(0, 10^2)
    #   sigma ~ HalfNormal(10^2)
    # -------------------------------------------------------------------------
    np.random.seed(42)
    n_obs = 100
    # Simulate weights (kg)
    weights = np.random.uniform(40.0, 90.0, size=n_obs)
    w_centered = weights - np.mean(weights)

    true_alpha = 175.0   # mean height (cm)
    true_beta = 0.90     # cm per kg
    true_sigma = 5.0     # residual noise (cm)

    heights = true_alpha + true_beta * w_centered + np.random.normal(0, true_sigma, size=n_obs)

    print(f"\nSimulated {n_obs} adult observations:")
    print(f"  True alpha: {true_alpha:.2f} cm")
    print(f"  True beta:  {true_beta:.2f} cm/kg")
    print(f"  True sigma: {true_sigma:.2f} cm")

    # Define log-posterior function
    def log_posterior(theta):
        alpha, beta, sigma = theta
        if sigma <= 0.0:
            return -np.inf

        # Log priors
        lp_alpha = -0.5 * ((alpha - 175.0) / 20.0) ** 2
        lp_beta = -0.5 * (beta / 10.0) ** 2
        lp_sigma = -0.5 * (sigma / 10.0) ** 2
        log_prior = lp_alpha + lp_beta + lp_sigma

        # Log likelihood
        mu = alpha + beta * w_centered
        residual = heights - mu
        log_lik = -n_obs * np.log(sigma) - np.sum(residual ** 2) / (2.0 * sigma ** 2)

        return log_prior + log_lik

    # Fit Quadratic / Laplace Approximation
    quap = Quap.fit(
        log_posterior_fn=log_posterior,
        start_params={'alpha': 170.0, 'beta': 0.0, 'sigma': 10.0},
        bounds={'alpha': (100.0, 250.0), 'beta': (-10.0, 10.0), 'sigma': (0.1, 50.0)}
    )

    print("\n--- Laplace Approximation Summary Table ---")
    summary_df = quap.summary(hdi=0.95)
    print(summary_df.to_string())

    print("\n--- Posterior Covariance Matrix ---")
    print(quap.cov.round(4).to_string())

    # Extract 1D marginal distributions
    beta_dist = quap.marginal('beta')
    print(f"\n--- Marginal Distribution for Beta ---")
    print(f"Distribution: {beta_dist}")
    print(f"Mean:         {beta_dist.mean():.4f}")
    print(f"Std Dev:      {beta_dist.std():.4f}")
    print(f"95% CI/HDI:   {beta_dist.interval(0.95)}")
    print(f"P(beta > 0):  {(1.0 - beta_dist.cdf().at(0.0)):.4%}")

    # Draw posterior samples
    samples = quap.sample(size=1000, seed=42)
    print(f"\nDrew {len(samples)} posterior samples:")
    print(samples.head(3).round(3).to_string())

    # -------------------------------------------------------------------------
    # 2. Visualizations: Forest Plot & Marginal Density
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    quap.plot_forest(
        ax=axes[0],
        hdi=0.95,
        title="Laplace Approximation: Parameter 95% HDIs"
    )

    quap.plot_marginal(
        param='beta',
        hdi=0.95,
        ax=axes[1]
    )

    plt.tight_layout()
    print("\nPlots generated successfully.")


if __name__ == "__main__":
    main()
