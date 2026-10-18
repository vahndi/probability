import matplotlib.pyplot as plt
import numpy as np

from probability.distributions import GridDistribution1d


def main():
    print("=" * 70)
    print("1D GRID APPROXIMATION & NON-CONJUGATE BAYESIAN INFERENCE")
    print("=" * 70)

    # 1. Setting up a non-conjugate prior for conversion rate p in [0, 1]
    # Suppose domain knowledge suggests p is likely around 0.15 or 0.35 (e.g. two distinct segments),
    # forming a bimodal prior that standard Beta conjugate priors cannot represent.
    grid = np.linspace(0.001, 0.999, 1000)

    # Define bimodal prior density: mixture of two normal-like components
    prior_density = (
        0.6 * np.exp(-0.5 * ((grid - 0.15) / 0.04) ** 2)
        + 0.4 * np.exp(-0.5 * ((grid - 0.35) / 0.05) ** 2)
    )
    prior_a = GridDistribution1d.from_function(lambda p: np.interp(p, grid, prior_density), grid)
    print("\n--- Non-Conjugate Prior Summary (Variant A) ---")
    print(f"Mean:              {prior_a.mean():.4f}")
    print(f"Std Dev:           {prior_a.std():.4f}")
    print(f"Mode:              {prior_a.mode():.4f}")
    print(f"Median:            {prior_a.median():.4f}")
    print(f"95% Equal-Tail CI: [{prior_a.credible_interval(0.95)[0]:.4f}, {prior_a.credible_interval(0.95)[1]:.4f}]")
    print(f"95% Shortest HDI:  [{prior_a.hdi(0.95)[0]:.4f}, {prior_a.hdi(0.95)[1]:.4f}]")

    # 2. Bayesian Updating with Binomial Data
    # Variant A observes 24 conversions out of 100 trials
    post_a = prior_a.update_binomial(k=24, n=100)
    print("\n--- Posterior Summary (Variant A: 24 / 100) ---")
    print(f"Posterior Mean:    {post_a.mean():.4f}")
    print(f"Posterior Std:     {post_a.std():.4f}")
    print(f"Posterior Mode:    {post_a.mode():.4f}")
    print(f"Posterior 95% HDI: [{post_a.hdi(0.95)[0]:.4f}, {post_a.hdi(0.95)[1]:.4f}]")

    # Variant B: standard uniform prior, observes 38 conversions out of 110 trials
    prior_b = GridDistribution1d.uniform(low=0.001, high=0.999, num_points=1000)
    post_b = prior_b.update_binomial(k=38, n=110)
    print("\n--- Posterior Summary (Variant B: 38 / 110) ---")
    print(f"Posterior Mean:    {post_b.mean():.4f}")
    print(f"Posterior Std:     {post_b.std():.4f}")
    print(f"Posterior Mode:    {post_b.mode():.4f}")
    print(f"Posterior 95% HDI: [{post_b.hdi(0.95)[0]:.4f}, {post_b.hdi(0.95)[1]:.4f}]")

    # 3. Exact Probability Comparisons
    # Calculate P(B > A) directly across the joint grid distribution
    prob_b_beats_a = post_b > post_a
    print(f"\nProbability Variant B > Variant A: {prob_b_beats_a:.4%}")

    # 4. Distribution Arithmetic (Difference Distribution)
    diff = post_b - post_a
    print("\n--- Difference Distribution (Variant B - Variant A) ---")
    print(f"Mean Difference:   {diff.mean():.4f}")
    print(f"Std of Diff:       {diff.std():.4f}")
    print(f"95% HDI of Diff:   [{diff.hdi(0.95)[0]:.4f}, {diff.hdi(0.95)[1]:.4f}]")
    print(f"P(Difference > 0): {(diff > 0.0):.4%}")

    # 5. Arbitrary Likelihood Updating with Custom Function
    # Suppose we have Poisson count observations [3, 4, 2, 5, 3] and an exponential decay prior
    rate_grid = np.linspace(0.1, 15.0, 1500)
    # Custom prior: e.g. half-normal-like prior
    prior_rate = GridDistribution1d.from_function(lambda lam: np.exp(-0.5 * (lam / 4.0) ** 2), rate_grid)
    # Update with Poisson data
    counts = [3, 4, 2, 5, 3]
    post_rate = prior_rate.update_poisson(k=counts)
    print("\n--- Poisson Rate Estimation ---")
    print(f"Observed counts:   {counts} (sum={sum(counts)}, n={len(counts)})")
    print(f"Rate Mean:         {post_rate.mean():.4f}")
    print(f"Rate 95% HDI:      [{post_rate.hdi(0.95)[0]:.4f}, {post_rate.hdi(0.95)[1]:.4f}]")

    # 6. Plotting with Shaded Credible & HDI Intervals
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    prior_a.plot(ax=axes[0], label="Prior A (Bimodal)", color="gray", alpha=0.6)
    post_a.plot(ax=axes[0], label="Posterior A", hdi=0.95, color="steelblue", title="Variant A: Prior vs. Posterior")
    axes[0].legend()

    diff.plot(ax=axes[1], hdi=0.95, color="darkgreen", title="Difference Distribution (B - A)")
    axes[1].axvline(0.0, color="red", linestyle="--", alpha=0.7, label="Zero Difference")
    axes[1].legend()

    plt.tight_layout()
    print("\nPlots generated successfully.")


if __name__ == "__main__":
    main()
