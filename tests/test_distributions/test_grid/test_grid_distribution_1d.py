from unittest import TestCase
from matplotlib.axes import Axes
import numpy as np
import pandas as pd

from probability.distributions.grid.grid_distribution_1d import (
    GridDistribution1d,
    BayesianGrid1d
)
from probability.distributions.continuous.beta import Beta
from probability.distributions.continuous.normal import Normal


class TestGridDistribution1d(TestCase):

    def setUp(self):
        self.grid = np.linspace(0, 10, 101)
        self.unif = GridDistribution1d.uniform(low=0.0, high=10.0, num_points=101)

    def test_initialization(self):
        # Uniform
        self.assertEqual(len(self.unif), 101)
        self.assertAlmostEqual(self.unif.probs.sum(), 1.0)
        self.assertAlmostEqual(self.unif.mean(), 5.0)

        # Direct probs
        probs = np.ones(11)
        dist = GridDistribution1d(grid=range(11), probs=probs)
        self.assertAlmostEqual(dist.probs.sum(), 1.0)
        self.assertAlmostEqual(dist.mean(), 5.0)

        # Alias
        bg = BayesianGrid1d.uniform(0, 1, 51)
        self.assertIsInstance(bg, GridDistribution1d)

    def test_from_distribution(self):
        # Beta(2, 5) on [0, 1]
        grid = np.linspace(0, 1, 1001)
        dist = GridDistribution1d.from_distribution(Beta(2, 5), grid=grid)
        # Analytical mean of Beta(2, 5) is 2/7 ≈ 0.2857
        self.assertAlmostEqual(dist.mean(), 2.0 / 7.0, places=2)
        # Mode of Beta(2, 5) is (2-1)/(2+5-2) = 1/5 = 0.20
        self.assertAlmostEqual(dist.mode(), 0.20, places=2)

    def test_from_function(self):
        # Gaussian unnormalized function exp(-0.5 * (x - 3)^2)
        grid = np.linspace(-2, 8, 101)
        dist = GridDistribution1d.from_function(lambda x: np.exp(-0.5 * (x - 3.0) ** 2), grid=grid)
        self.assertAlmostEqual(dist.mean(), 3.0, places=2)
        self.assertAlmostEqual(dist.mode(), 3.0, places=2)
        self.assertAlmostEqual(dist.std(), 1.0, places=1)

    def test_summary_metrics(self):
        # Triangular distribution on {1, 2, 3} with probs [0.25, 0.5, 0.25]
        dist = GridDistribution1d(grid=[1.0, 2.0, 3.0], probs=[0.25, 0.5, 0.25])
        self.assertAlmostEqual(dist.mean(), 2.0)
        self.assertAlmostEqual(dist.mode(), 2.0)
        self.assertAlmostEqual(dist.median(), 2.0)
        # Var = 0.25*(1-2)^2 + 0.5*0 + 0.25*(3-2)^2 = 0.5
        self.assertAlmostEqual(dist.var(), 0.5)
        self.assertAlmostEqual(dist.std(), np.sqrt(0.5))

        # Quantiles
        self.assertAlmostEqual(dist.quantile(0.25), 1.0)
        self.assertAlmostEqual(dist.quantile(0.75), 2.0)

    def test_intervals(self):
        grid = np.linspace(-5, 5, 1001)
        dist = GridDistribution1d.from_distribution(Normal(0, 1), grid=grid)

        # 95% Credible interval for Standard Normal ≈ [-1.96, 1.96]
        ci_low, ci_high = dist.credible_interval(0.95)
        self.assertAlmostEqual(ci_low, -1.96, places=1)
        self.assertAlmostEqual(ci_high, 1.96, places=1)

        # HDI for symmetric unimodal distribution matches CI
        hdi_low, hdi_high = dist.hdi(0.95)
        self.assertAlmostEqual(hdi_low, -1.96, places=1)
        self.assertAlmostEqual(hdi_high, 1.96, places=1)

    def test_bayesian_update_binomial(self):
        # Uniform prior on p in [0, 1]
        prior = GridDistribution1d.uniform(0.0, 1.0, num_points=1001)
        # Observe 7 heads in 10 flips
        post = prior.update_binomial(k=7, n=10)

        # Conjugate Beta(1+7, 1+3) = Beta(8, 4)
        # Mean = 8 / 12 = 0.6667, Mode = 7 / 10 = 0.70
        self.assertAlmostEqual(post.mean(), 8.0 / 12.0, places=2)
        self.assertAlmostEqual(post.mode(), 0.70, places=2)

    def test_bayesian_update_poisson(self):
        prior = GridDistribution1d.uniform(0.1, 20.0, num_points=200)
        # Counts: [4, 5, 6], mean = 5.0
        post = prior.update_poisson(k=[4, 5, 6])
        self.assertAlmostEqual(post.mode(), 5.0, places=1)

    def test_bayesian_update_normal(self):
        prior = GridDistribution1d.uniform(-10.0, 10.0, num_points=401)
        # Data with true mean 2.5 and known sigma = 1.0
        post = prior.update_normal(data=[2.4, 2.6, 2.5], sigma=1.0)
        self.assertAlmostEqual(post.mode(), 2.5, places=2)

    def test_comparisons(self):
        # dist1 centered at 3, dist2 centered at 7
        g = np.linspace(0, 10, 101)
        d1 = GridDistribution1d.from_distribution(Normal(3, 1), grid=g)
        d2 = GridDistribution1d.from_distribution(Normal(7, 1), grid=g)

        # Scalar comparisons
        self.assertAlmostEqual(d1 > 3.0, 0.5, places=1)
        self.assertAlmostEqual(d1 < 3.0, 0.5, places=1)
        self.assertGreater(float(d2 > 5.0), 0.95)

        # Distribution vs Distribution comparisons
        # P(d2 > d1) should be ≈ 1.0
        prob_d2_gt_d1 = d2 > d1
        self.assertIsInstance(prob_d2_gt_d1, float)
        self.assertGreater(prob_d2_gt_d1, 0.95)

        prob_d1_gt_d2 = d1 > d2
        self.assertIsInstance(prob_d1_gt_d2, float)
        self.assertLess(prob_d1_gt_d2, 0.05)

    def test_arithmetic(self):
        # Shifts
        d = GridDistribution1d(grid=[1.0, 2.0, 3.0], probs=[0.2, 0.5, 0.3])
        d_shifted = d + 5.0
        self.assertAlmostEqual(d_shifted.mean(), 7.1)

        d_scaled = 2.0 * d
        self.assertAlmostEqual(d_scaled.mean(), 4.2)

        d_neg = -d
        self.assertAlmostEqual(d_neg.mean(), -2.1)

        # Distribution addition: d + d
        d_sum = d + d
        self.assertAlmostEqual(d_sum.mean(), 4.2, places=1)

    def test_sampling_and_plotting(self):
        grid = np.linspace(-3, 3, 101)
        dist = GridDistribution1d.from_distribution(Normal(0, 1), grid=grid)

        samples = dist.sample(num_samples=50, random_state=42)
        self.assertEqual(len(samples), 50)

        rvs = dist.rvs(num_samples=25, random_state=42)
        self.assertIsInstance(rvs, pd.Series)
        self.assertEqual(len(rvs), 25)

        ax = dist.plot(ci=0.95, hdi=0.95)
        self.assertIsInstance(ax, Axes)
