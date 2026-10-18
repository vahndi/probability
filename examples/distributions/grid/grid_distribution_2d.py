import matplotlib.pyplot as plt
import numpy as np

from probability.distributions import GridDistribution2d


def main():
    print("=" * 75)
    print("2D GRID APPROXIMATION & JOINT BAYESIAN INFERENCE")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # 1. Joint Inference for Normal Parameters (μ, σ)
    # Classic problem from Statistical Rethinking (Chapter 4):
    # Estimating height distribution (mean μ and standard deviation σ)
    # -------------------------------------------------------------------------
    np.random.seed(42)
    true_mu = 175.0  # true height mean (cm)
    true_sigma = 8.0  # true height std (cm)
    n_obs = 50
    heights = np.random.normal(loc=true_mu, scale=true_sigma, size=n_obs)

    print(f"\n--- Observed Data (N = {n_obs}) ---")
    print(f"Sample Mean:       {np.mean(heights):.3f} cm")
    print(f"Sample Std:        {np.std(heights, ddof=1):.3f} cm")

    # Define 2D grid prior over mu in [165, 185] and sigma in [4, 14]
    prior = GridDistribution2d.uniform(
        x_low=165.0, x_high=185.0, x_num=100,
        y_low=4.0, y_high=14.0, y_num=100,
        x_name="mu", y_name="sigma"
    )

    # Bayesian update with Normal data
    posterior = prior.update_normal(heights)

    post_mode_mu, post_mode_sigma = posterior.mode()
    post_mean_mu, post_mean_sigma = posterior.mean()

    print("\n--- Joint Posterior Estimates ---")
    print(f"Joint Mode (MAP):  mu = {post_mode_mu:.3f}, sigma = {post_mode_sigma:.3f}")
    print(f"Joint Mean:        mu = {post_mean_mu:.3f}, sigma = {post_mean_sigma:.3f}")
    print(f"Correlation:       r(mu, sigma) = {posterior.corr():.4f}")

    cov = posterior.cov()
    print("2x2 Covariance Matrix:")
    print(f"  [[Var(mu):    {cov[0, 0]:.4f}, Cov(mu, sig): {cov[0, 1]:.4f}],")
    print(f"   [Cov(mu, sig):{cov[1, 0]:.4f}, Var(sigma):   {cov[1, 1]:.4f}]]")

    # -------------------------------------------------------------------------
    # 2. Extracting Component Marginals & Conditionals
    # -------------------------------------------------------------------------
    marg_mu = posterior.marginal_x()
    marg_sigma = posterior.marginal_y()

    print("\n--- Marginal Distributions (GridDistribution1d) ---")
    print(f"Marginal mu:    Mean={marg_mu.mean():.3f}, 95% HDI={marg_mu.hdi(0.95)}")
    print(f"Marginal sigma: Mean={marg_sigma.mean():.3f}, 95% HDI={marg_sigma.hdi(0.95)}")

    # Conditional slice: P(mu | sigma = 8.0)
    cond_mu_given_sig = posterior.conditional_x(given_y=8.0)
    print(f"Conditional P(mu | sigma ≈ 8.0): Mean={cond_mu_given_sig.mean():.3f}, Std={cond_mu_given_sig.std():.3f}")

    # -------------------------------------------------------------------------
    # 3. Joint Inference for 2D Logistic Regression (Intercept α, Slope β)
    # -------------------------------------------------------------------------
    doses = np.array([-2.0, -1.0, 0.0, 1.0, 2.0] * 8)
    # Ground truth: alpha = 0.2, beta = 1.4
    prob_response = 1.0 / (1.0 + np.exp(-(0.2 + 1.4 * doses)))
    cured = (np.random.uniform(size=len(doses)) < prob_response).astype(int)

    prior_logit = GridDistribution2d.uniform(
        x_low=-2.0, x_high=2.0, x_num=60,
        y_low=-0.5, y_high=3.0, y_num=60,
        x_name="alpha (intercept)", y_name="beta (slope)"
    )
    post_logit = prior_logit.update_logistic_regression(x_data=doses, y_binary=cured)
    mode_alpha, mode_beta = post_logit.mode()
    print("\n--- Logistic Regression Dose-Response ---")
    print(f"Posterior Mode:    alpha={mode_alpha:.3f}, beta={mode_beta:.3f}")
    print(f"P(beta > 0):       {(post_logit.marginal_y() > 0.0):.4%}")

    # -------------------------------------------------------------------------
    # 4. Visualizations: Contour with 95% HDR & Heatmap
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Height posterior: Contour plot with 95% HDR boundary line
    posterior.plot_contour(
        ax=axes[0],
        levels=12,
        fill=True,
        cmap="Blues",
        hdr=0.95,
        title="Joint Posterior P(mu, sigma | Data) with 95% HDR"
    )

    # Logistic regression: Heatmap
    post_logit.plot_heatmap(
        ax=axes[1],
        cmap="viridis",
        title="Logistic Regression Joint Posterior (alpha, beta)"
    )

    plt.tight_layout()
    print("\nPlots generated successfully.")


if __name__ == "__main__":
    main()
