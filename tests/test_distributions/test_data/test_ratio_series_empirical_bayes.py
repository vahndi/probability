import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from probability.distributions import Ratio, RatioSeries, Normal
from probability.distributions.continuous.normal_series import NormalSeries


@pytest.fixture
def multi_group_ratios():
    """
    Simulated continuous grouped ratio/metric data from true Normal grand prior
    mu_0 = 75.0, tau_0 = 5.0, with measurement noise sigma_j = 10.0 across 12 groups
    with varying sample sizes.
    """
    np.random.seed(42)
    true_mu0 = 75.0
    true_tau0 = 5.0
    within_sigma = 10.0
    n_groups = 12

    group_means = np.random.normal(loc=true_mu0, scale=true_tau0, size=n_groups)
    sample_sizes = [2, 3, 5, 8, 12, 20, 35, 50, 80, 150, 250, 400]

    series_dict = {}
    for i, (grp_mean, n) in enumerate(zip(group_means, sample_sizes)):
        obs = np.random.normal(loc=grp_mean, scale=within_sigma, size=n)
        series_dict[f'Group_{i+1}'] = Ratio(pd.Series(obs))

    return RatioSeries(series_dict)


class TestRatioSeriesEmpiricalBayes:

    def test_empirical_bayes_prior_mle(self, multi_group_ratios):
        prior = multi_group_ratios.empirical_bayes_prior(method='mle')
        assert isinstance(prior, Normal)
        assert prior.sigma > 0.0
        # Grand mean should be close to true mu_0 = 75.0
        assert np.isclose(prior.mu, 75.0, atol=3.0)

    def test_empirical_bayes_prior_moments(self, multi_group_ratios):
        prior = multi_group_ratios.empirical_bayes_prior(method='moments')
        assert isinstance(prior, Normal)
        assert prior.sigma > 0.0
        assert np.isclose(prior.mu, 75.0, atol=3.0)

    def test_shrunk_posteriors_shrinkage_behavior(self):
        # 3 groups with explicit Normal prior mu_0=70.0, tau_0=5.0
        ratio_dict = {
            'Large_Sample': Ratio(pd.Series([90.0] * 1000)),
            'Small_Outlier_High': Ratio(pd.Series([110.0, 114.0])),   # mean 112, small n=2
            'Small_Outlier_Low': Ratio(pd.Series([30.0, 34.0])),      # mean 32, small n=2
        }
        r_series = RatioSeries(ratio_dict)

        prior = Normal(mu=70.0, sigma=5.0)
        shrunk = r_series.shrunk_posteriors(prior=prior)
        assert isinstance(shrunk, NormalSeries)

        # 1. Large sample mean ~90 retains its sample mean
        large_shrunk = shrunk['Large_Sample'].mu
        assert np.isclose(large_shrunk, 90.0, atol=0.2)

        # 2. Small high outlier mean 112 is shrunk substantially toward prior mean 70
        high_shrunk = shrunk['Small_Outlier_High'].mu
        assert high_shrunk < 112.0
        assert high_shrunk > 70.0

        # 3. Small low outlier mean 32 is shrunk substantially upward toward prior mean 70
        low_shrunk = shrunk['Small_Outlier_Low'].mu
        assert low_shrunk > 32.0
        assert low_shrunk < 70.0

    def test_shrinkage_summary(self, multi_group_ratios):
        summary = multi_group_ratios.shrinkage_summary(hdi=0.95)
        assert 'n' in summary.columns
        assert 'raw_mean' in summary.columns
        assert 'raw_se' in summary.columns
        assert 'shrunk_mean' in summary.columns
        assert 'shrunk_std' in summary.columns
        assert '95% HDI lower' in summary.columns
        assert '95% HDI upper' in summary.columns
        assert 'weight_data' in summary.columns

        # Verify weights increase with sample size n
        sorted_by_n = summary.sort_values('n')
        weights = sorted_by_n['weight_data'].values
        assert weights[-1] > weights[0]
        assert np.all(weights >= 0.0) and np.all(weights <= 1.0)

    def test_plot_shrinkage(self, multi_group_ratios):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        ax1 = multi_group_ratios.plot_shrinkage(ax=axes[0], order_by='raw_mean')
        ax2 = multi_group_ratios.plot_shrinkage(ax=axes[1], order_by='n')
        assert ax1 is axes[0]
        assert ax2 is axes[1]
        plt.close(fig)

    def test_error_handling(self):
        r_series_1 = RatioSeries({'OnlyOne': Ratio(pd.Series([10.0, 12.0, 15.0]))})
        with pytest.raises(ValueError, match="at least 2 categories"):
            r_series_1.empirical_bayes_prior()

        r_series_2 = RatioSeries({
            'G1': Ratio(pd.Series([10.0, 11.0])),
            'G2': Ratio(pd.Series([20.0, 22.0]))
        })
        with pytest.raises(ValueError, match="Unknown method"):
            r_series_2.empirical_bayes_prior(method='invalid')
