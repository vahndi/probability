from unittest import TestCase

from numpy import arange, inf
from pandas import Series
from scipy.stats import moment
from scipy.stats.distributions import norm

from probability.distributions.continuous.continuous_data import ContinuousData


class TestContinuousData(TestCase):

    def setUp(self) -> None:

        self.data = Series(norm(5, 1).rvs(1_000))
        self.data2 = Series(norm(10, 1).rvs(1_000))
        self.dist = ContinuousData(self.data)
        self.dist2 = ContinuousData(self.data2)

    def test_lower_bound(self):

        self.assertEqual(self.data.min(), self.dist.lower_bound)

    def test_upper_bound(self):

        self.assertEqual(self.data.max(), self.dist.upper_bound)

    def test_mean(self):

        self.assertEqual(self.data.mean(), self.dist.mean())

    def test_median(self):

        self.assertEqual(self.data.median(), self.dist.median())

    def test_moment(self):

        for n in range(1, 6):
            self.assertAlmostEqual(
                moment(self.data, n), self.dist.moment(n), 5
            )

    def test_std(self):

        self.assertAlmostEqual(self.data.std(), self.dist.std(), 5)

    def test_var(self):

        self.assertAlmostEqual(self.data.var(), self.dist.var(), 5)

    def test_interval(self):

        for conf in arange(0.05, 1.0, 0.95):
            lower_pct = 0.5 - conf / 2
            upper_pct = 0.5 + conf / 2
            lower_val, upper_val = self.data.quantile([lower_pct, upper_pct])
            self.assertTupleEqual((lower_val, upper_val),
                                  self.dist.interval(conf))

    def test_support(self):

        self.assertTupleEqual(
            (-inf, inf), self.dist.support()
        )

    def test_rvs(self):

        n_samples = 100_000
        expected = self.data.sample(
            n=n_samples, replace=True, random_state=0)
        actual = self.dist.rvs(num_samples=n_samples, random_state=0)
        self.assertAlmostEqual(expected.mean(), actual.mean(), 1)
        self.assertAlmostEqual(expected.var(), actual.var(), 1)
        self.assertAlmostEqual(expected.kurt(), actual.kurt(), 1)
