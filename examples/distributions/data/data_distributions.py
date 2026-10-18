"""
Example demonstrating empirical Data distributions (Ratio, Boolean, Nominal),
sampling statistics (SE, mean_dist, CI, two-sample tests, sample size),
Cohen's d (normalized_diff), and quantile/pipe filtering.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from probability.distributions import Ratio, Boolean, Nominal
from probability.plotting import new_axes


def demonstrate_ratio_stats():
    print("=" * 60)
    print("1. Ratio Distribution & Sampling Statistics")
    print("=" * 60)

    np.random.seed(42)
    # Control vs. Treatment group conversion revenue
    control_vals = np.random.gamma(shape=5.0, scale=10.0, size=200)
    treat_vals = np.random.gamma(shape=5.5, scale=10.0, size=250)

    control = Ratio(pd.Series(control_vals, name='Control'))
    treat = Ratio(pd.Series(treat_vals, name='Treatment'))

    print(f"Control: n={len(control)}, mean={control.mean():.2f}, std={control.std():.2f}")
    print(f"Control SE (Standard Error): {control.se():.3f}")
    print(f"Control 95% Confidence Interval: {control.ci(0.95)}")
    print(f"Control sampling distribution of mean: {control.mean_dist()}")

    print(f"\nTreatment: n={len(treat)}, mean={treat.mean():.2f}, std={treat.std():.2f}")
    print(f"Treatment SE: {treat.se():.3f}")
    print(f"Treatment 95% CI: {treat.ci(0.95)}")

    # 2. Two-Sample Difference in Means & Hypothesis Testing
    print("\n" + "=" * 60)
    print("2. Two-Sample Difference in Means & Hypothesis Testing")
    print("=" * 60)
    diff_dist = treat.mean_diff(control)
    print(f"Difference in Means Distribution (Treat - Control): {diff_dist}")
    print(f"Estimated Effect: {treat.mean() - control.mean():.2f}")
    print(f"95% CI for difference: {treat.mean_diff_ci(control, percent=0.95)}")

    t = treat.t_stat(control)
    p = treat.p_value(control)
    print(f"Hypothesis Test: t = {t:.3f}, p = {p:.4f}")
    print(f"Reject null at 5%? {'Yes' if p < 0.05 else 'No'}")

    # Cohen's d effect size
    d_pooled = treat.cohens_d(control, pooled=True)
    d_unpooled = treat.cohens_d(control, pooled=False)
    print(f"Cohen's d (pooled): {d_pooled:.4f}")
    print(f"Normalized Difference (unpooled): {d_unpooled:.4f}")

    # Required sample size for detecting MDE
    mde = 3.0
    req_n = treat.sample_size_for_effect(mde=mde, power=0.80, alpha=0.05)
    print(f"Required sample size per group for MDE = {mde:.1f} (power=80%, alpha=5%): {req_n:.0f}")

    return control, treat, diff_dist


def demonstrate_filtering_and_conditioning():
    print("\n" + "=" * 60)
    print("3. Filtering, Indicators & Pipe Conditioning")
    print("=" * 60)

    np.random.seed(123)
    values = pd.Series(np.random.normal(loc=50, scale=15, size=500), name='Score')
    dist = Ratio(values)

    print(f"Original n={len(dist)}, mean={dist.mean():.2f}, range=({dist.min():.2f}, {dist.max():.2f})")

    # Quantile filtering (e.g. trim extreme 5% outliers on both ends)
    trimmed = dist.between_quantiles(0.05, 0.95)
    print(f"Trimmed (5th to 95th quantile): n={len(trimmed)}, range=({trimmed.min():.2f}, {trimmed.max():.2f})")

    # Indicator series & boolean operations
    is_high = dist.gt(60)
    is_low = dist.lt(40)
    is_mid = (~is_high).logical_and(~is_low)
    print(f"Scores > 60: {(is_high.data == True).sum()} ({is_high.mean():.1%})")
    print(f"Scores < 40: {(is_low.data == True).sum()} ({is_low.mean():.1%})")
    print(f"Scores in [40, 60]: {(is_mid.data == True).sum()} ({is_mid.mean():.1%})")

    # Pipe syntax conditioning: conditioning on Boolean splits into a RatioSeries
    split_scores = dist | is_high
    print(f"Conditioned via pipe (Score | is_high):\n  Means by category:\n{split_scores.mean()}")
    high_group = split_scores[True]
    print(f"  True group (Score > 60): n={len(high_group)}, mean={high_group.mean():.2f}")

    between_dist = dist.between(45, 55)
    print(f"Conditioned via between (45 <= Score <= 55): n={len(between_dist)}, mean={between_dist.mean():.2f}")


def plot_distributions(control, treat, diff_dist):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Control vs. Treatment density plots
    control.plot_kde(ax=axes[0], color='tab:blue')
    treat.plot_kde(ax=axes[0], color='tab:orange')
    axes[0].set_title("Empirical Ratio KDE Distributions (Control vs Treat)")

    # Sampling distribution with shaded 95% Confidence Interval
    diff_dist.plot(ci=0.95, ax=axes[1], color='tab:green')
    axes[1].set_title(f"Difference in Means Sampling Dist (Treat - Control)\n95% CI: [{diff_dist.interval(0.95)[0]:.2f}, {diff_dist.interval(0.95)[1]:.2f}]")
    axes[1].axvline(0, color='gray', linestyle='--', alpha=0.7, label='Null (diff=0)')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    control, treat, diff_dist = demonstrate_ratio_stats()
    demonstrate_filtering_and_conditioning()
    plot_distributions(control, treat, diff_dist)
