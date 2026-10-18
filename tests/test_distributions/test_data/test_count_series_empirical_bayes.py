import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from probability.distributions import Count, CountSeries, Gamma
from probability.distributions.continuous.gamma_series import GammaSeries
from probability.distributions.discrete.negative_binomial import NegativeBinomial


@pytest.fixture
def multi_group_counts():
    """
    Simulated grouped count data from true Gamma(alpha=6.0, beta=2.0)
    (true mean rate lambda = 3.0) across 15 categories with diverse exposure sizes.
    """
    np.random.seed(42)
    true_alpha, true_beta = 6.0, 2.0
    n_groups = 15

    # True group rates drawn from Gamma prior
    true_lambdas = np.random.gamma(shape=true_alpha, scale=1.0 / true_beta, size=n_groups)
    exposures = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300]

    series_dict = {}
    for i, (lam, n) in enumerate(zip(true_lambdas, exposures)):
        # Generate n Poisson observations with rate lam
        counts = np.random.poisson(lam=lam, size=n)
        series_dict[f'Group_{i+1}'] = Count(pd.Series(counts))

    return CountSeries(series_dict)


class TestCountSeriesEmpiricalBayes:

    def test_empirical_bayes_prior_mle(self, multi_group_counts):
        prior = multi_group_counts.empirical_bayes_prior(method='mle')
        assert isinstance(prior, Gamma)
        assert prior.alpha > 0.0
        assert prior.beta > 0.0

        # Mean rate should be close to true mean 3.0 (alpha=6, beta=2)
        assert np.isclose(prior.mean(), 3.0, atol=0.5)

    def test_empirical_bayes_prior_moments(self, multi_group_counts):
        prior = multi_group_counts.empirical_bayes_prior(method='moments')
        assert isinstance(prior, Gamma)
        assert prior.alpha > 0.0
        assert prior.beta > 0.0
        assert np.isclose(prior.mean(), 3.0, atol=0.7)

    def test_shrunk_posteriors_shrinkage_behavior(self):
        # 3 groups with explicit Gamma prior
        count_dict = {
            'Large_Normal': Count(pd.Series([3] * 500)),
            'Small_Extreme_High': Count(pd.Series([12])),
            'Small_Extreme_Low': Count(pd.Series([0])),
        }
        c_series = CountSeries(count_dict)

        prior = Gamma(alpha=6.0, beta=2.0)  # prior mean = 3.0
        shrunk = c_series.shrunk_posteriors(prior=prior)
        assert isinstance(shrunk, GammaSeries)

        # 1. Extreme high rate 12.0 is shrunk heavily toward prior mean 3.0 (to 18/3 = 6.0)
        high_shrunk = shrunk['Small_Extreme_High'].mean()
        assert np.isclose(high_shrunk, 6.0, atol=0.01)

        # 2. Extreme low rate 0.0 is shrunk upward toward prior mean 3.0 (to 6/3 = 2.0)
        low_shrunk = shrunk['Small_Extreme_Low'].mean()
        assert np.isclose(low_shrunk, 2.0, atol=0.01)

        # 3. Large normal rate (500 obs with 1500 events) stays near 3.0: (1500+6)/(500+2) = 1506/502 = 3.0
        large_shrunk = shrunk['Large_Normal'].mean()
        assert np.isclose(large_shrunk, 3.0, atol=0.01)

    def test_predictive_distributions(self, multi_group_counts):
        preds = multi_group_counts.predictive_distributions(time_exposure=1.0)
        assert len(preds) == len(multi_group_counts.keys())
        for cat, dist in preds.items():
            assert isinstance(dist, NegativeBinomial)
            assert dist.r > 0.0
            assert 0.0 < dist.p < 1.0

    def test_shrinkage_summary(self, multi_group_counts):
        summary = multi_group_counts.shrinkage_summary(hdi=0.95)
        assert 'total_count' in summary.columns
        assert 'exposure' in summary.columns
        assert 'raw_rate' in summary.columns
        assert 'shrunk_mean' in summary.columns
        assert '95% HDI lower' in summary.columns
        assert '95% HDI upper' in summary.columns
        assert 'weight_data' in summary.columns

        # Verify weights increase with exposure
        sorted_by_n = summary.sort_values('exposure')
        weights = sorted_by_n['weight_data'].values
        assert weights[-1] > weights[0]
        assert np.all(weights >= 0.0) and np.all(weights <= 1.0)

    def test_plot_shrinkage(self, multi_group_counts):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        ax1 = multi_group_counts.plot_shrinkage(ax=axes[0], order_by='raw_rate')
        ax2 = multi_group_counts.plot_shrinkage(ax=axes[1], order_by='exposure')
        assert ax1 is axes[0]
        assert ax2 is axes[1]
        plt.close(fig)

    def test_error_handling(self):
        c_series_1 = CountSeries({'OnlyOne': Count(pd.Series([1, 2, 3]))})
        with pytest.raises(ValueError, match="at least 2 categories"):
            c_series_1.empirical_bayes_prior()

        c_series_2 = CountSeries({
            'G1': Count(pd.Series([1, 2])),
            'G2': Count(pd.Series([3, 4]))
        })
        with pytest.raises(ValueError, match="Unknown method"):
            c_series_2.empirical_bayes_prior(method='invalid')
