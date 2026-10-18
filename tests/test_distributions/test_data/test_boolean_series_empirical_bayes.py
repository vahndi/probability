import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from probability.distributions import Boolean, BooleanSeries, Beta
from probability.distributions.continuous.beta_series import BetaSeries


@pytest.fixture
def multi_group_data():
    """
    Simulated grouped conversion data with true underlying Beta(10, 40) distribution
    (grand mean = 0.20) across 20 categories with diverse sample sizes.
    """
    np.random.seed(42)
    true_alpha, true_beta = 10.0, 40.0
    n_groups = 20

    # Draw true group conversion rates
    true_ps = np.random.beta(true_alpha, true_beta, size=n_groups)

    # Varied sample sizes: some tiny (n=2, 5) and some large (n=500)
    sample_sizes = [2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 1, 4, 12, 250]

    series_dict = {}
    for i, (p, n) in enumerate(zip(true_ps, sample_sizes)):
        # Generate binomial outcomes
        k = np.random.binomial(n, p)
        outcomes = [True] * k + [False] * (n - k)
        series_dict[f'Group_{i+1}'] = Boolean(pd.Series(outcomes))

    return BooleanSeries(series_dict)


class TestBooleanSeriesEmpiricalBayes:

    def test_empirical_bayes_prior_mle(self, multi_group_data):
        prior = multi_group_data.empirical_bayes_prior(method='mle')
        assert isinstance(prior, Beta)
        assert prior.alpha > 0.0
        assert prior.beta > 0.0

        # Mean should be close to true grand mean 0.20 (10 / (10 + 40))
        assert np.isclose(prior.mean(), 0.20, atol=0.05)

    def test_empirical_bayes_prior_moments(self, multi_group_data):
        prior = multi_group_data.empirical_bayes_prior(method='moments')
        assert isinstance(prior, Beta)
        assert prior.alpha > 0.0
        assert prior.beta > 0.0
        assert np.isclose(prior.mean(), 0.20, atol=0.06)

    def test_shrunk_posteriors_shrinkage_behavior(self):
        # Create 3 synthetic groups:
        # Group A: large sample (n=1000, 200 successes -> 20%)
        # Group B: small sample extreme positive (n=2, 2 successes -> 100%)
        # Group C: small sample extreme zero (n=2, 0 successes -> 0%)
        # Group D: medium sample (n=100, 20 successes -> 20%)
        bool_dict = {
            'Large_Normal': Boolean(pd.Series([True] * 200 + [False] * 800)),
            'Small_Extreme_High': Boolean(pd.Series([True] * 2)),
            'Small_Extreme_Low': Boolean(pd.Series([False] * 2)),
            'Medium_Normal': Boolean(pd.Series([True] * 20 + [False] * 80)),
        }
        b_series = BooleanSeries(bool_dict)

        # Fit prior
        prior = b_series.empirical_bayes_prior()
        grand_mean = prior.mean()

        shrunk = b_series.shrunk_posteriors(prior=prior)
        assert isinstance(shrunk, BetaSeries)

        # 1. Small extreme high (raw 1.0) must be shrunk heavily towards grand mean
        high_shrunk_mean = shrunk['Small_Extreme_High'].mean()
        assert high_shrunk_mean < 0.60  # Pulled down from 1.00 significantly!

        # 2. Small extreme low (raw 0.0) must be shrunk heavily towards grand mean
        low_shrunk_mean = shrunk['Small_Extreme_Low'].mean()
        assert low_shrunk_mean > 0.05   # Pulled up from 0.00 significantly!

        # 3. Large normal (raw 0.20, n=1000) should stay virtually unshifted
        large_shrunk_mean = shrunk['Large_Normal'].mean()
        assert np.isclose(large_shrunk_mean, 0.20, atol=0.01)

    def test_shrinkage_summary(self, multi_group_data):
        summary = multi_group_data.shrinkage_summary(hdi=0.95)
        assert 'k' in summary.columns
        assert 'n' in summary.columns
        assert 'raw_mean' in summary.columns
        assert 'shrunk_mean' in summary.columns
        assert '95% HDI lower' in summary.columns
        assert '95% HDI upper' in summary.columns
        assert 'weight_data' in summary.columns

        # Verify weights increase monotonically with sample size n
        sorted_by_n = summary.sort_values('n')
        weights = sorted_by_n['weight_data'].values
        assert weights[-1] > weights[0]
        assert np.all(weights >= 0.0) and np.all(weights <= 1.0)

    def test_plot_shrinkage(self, multi_group_data):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        ax1 = multi_group_data.plot_shrinkage(ax=axes[0], order_by='raw_mean')
        ax2 = multi_group_data.plot_shrinkage(ax=axes[1], order_by='n')
        assert ax1 is axes[0]
        assert ax2 is axes[1]
        plt.close(fig)

    def test_error_handling(self):
        # Single category raises error
        b_series_1 = BooleanSeries({'OnlyOne': Boolean(pd.Series([True, False]))})
        with pytest.raises(ValueError, match="at least 2 categories"):
            b_series_1.empirical_bayes_prior()

        # Invalid method
        b_series_2 = BooleanSeries({
            'G1': Boolean(pd.Series([True, False])),
            'G2': Boolean(pd.Series([True, True]))
        })
        with pytest.raises(ValueError, match="Unknown method"):
            b_series_2.empirical_bayes_prior(method='invalid')
