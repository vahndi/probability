"""
Example demonstrating Multivariate Normal-Inverse-Wishart Conjugate Inference
(MVNormalInverseWishartConjugate) for joint unknown mean vector μ and covariance matrix Σ.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from probability.distributions.conjugate import MVNormalInverseWishartConjugate


def run_example():
    # 1. Simulate bivariate asset return data
    np.random.seed(42)
    n = 80
    true_mu = [0.05, 0.08]
    true_cov = [
        [0.04, 0.02],
        [0.02, 0.06]
    ]
    raw_data = np.random.multivariate_normal(mean=true_mu, cov=true_cov, size=n)
    df = pd.DataFrame(raw_data, columns=['Asset_A', 'Asset_B'])

    print("=" * 65)
    print("Multivariate Normal-Inverse-Wishart Conjugate Model")
    print("=" * 65)
    print(f"Sample data: n={n}")
    print(f"Sample mean:\n{df.mean()}")
    print(f"\nSample covariance:\n{df.cov()}")

    # 2. Fit analytical conjugate model
    model = MVNormalInverseWishartConjugate.infer_posterior(df)

    print("\n" + "-" * 65)
    print("Posterior Parameter Distributions:")
    print("-" * 65)
    post_mu = model.posterior_mu()
    print(f"Posterior mean vector mu ~ {post_mu}")
    print(f"  Asset_A marginal mu: {model.marginal_mu('Asset_A')}")
    print(f"  Asset_A 95% HDI: {model.marginal_mu('Asset_A').hdi(0.95)}")
    print(f"  Asset_B marginal mu: {model.marginal_mu('Asset_B')}")
    print(f"  Asset_B 95% HDI: {model.marginal_mu('Asset_B').hdi(0.95)}")

    post_sig = model.posterior_sigma()
    print(f"\nPosterior covariance Sigma ~ {post_sig}")
    print(f"Expected posterior covariance matrix:\n{post_sig.mean()}")
    print(f"Asset_A marginal variance sigma^2 ~ {model.marginal_sigma_sq('Asset_A')}")
    print(f"Asset_B marginal variance sigma^2 ~ {model.marginal_sigma_sq('Asset_B')}")

    # 3. Posterior predictive distribution for new observation
    pred = model.posterior_predictive()
    print(f"\nPosterior Predictive Distribution x_new ~ {pred}")

    # 4. Draw joint (mu, Sigma) samples from posterior
    mu_draws, sig_draws = model.sample_posterior(num_samples=500, random_state=42)
    print(f"\nPosterior draws: {len(mu_draws)} samples")
    print(f"Mean of sampled mu:\n{mu_draws.mean()}")

    # 5. Visual 2D plot with credible ellipses
    fig, ax = plt.subplots(figsize=(8, 6))
    model.plot(dim1='Asset_A', dim2='Asset_B', ax=ax)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_example()
