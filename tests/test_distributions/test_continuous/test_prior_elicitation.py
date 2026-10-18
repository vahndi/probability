from unittest.case import TestCase

import numpy as np

from probability.distributions import (
    Beta,
    ContinuousUniform,
    Gamma,
    InverseGamma,
    LogNormal,
    Normal,
)


class TestPriorElicitation(TestCase):

    def test_normal_from_interval(self):
        # 95% interval between -2 and 10
        norm = Normal.from_interval(-2.0, 10.0, prob=0.95)
        self.assertAlmostEqual(norm.cdf().at(-2.0), 0.025, places=4)
        self.assertAlmostEqual(norm.cdf().at(10.0), 0.975, places=4)
        lower, upper = norm.interval(0.95)
        self.assertAlmostEqual(lower, -2.0, places=4)
        self.assertAlmostEqual(upper, 10.0, places=4)

    def test_normal_from_quantiles(self):
        norm = Normal.from_quantiles(q1=0.10, v1=5.0, q2=0.90, v2=15.0)
        self.assertAlmostEqual(norm.cdf().at(5.0), 0.10, places=4)
        self.assertAlmostEqual(norm.cdf().at(15.0), 0.90, places=4)

    def test_normal_from_mean_and_std(self):
        norm = Normal.from_mean_and_std(mean=12.5, std=3.2)
        self.assertAlmostEqual(norm.mean(), 12.5, places=5)
        self.assertAlmostEqual(norm.std(), 3.2, places=5)

    def test_log_normal_from_interval(self):
        # 95% interval between 10 and 100
        ln = LogNormal.from_interval(10.0, 100.0, prob=0.95)
        self.assertAlmostEqual(ln.cdf().at(10.0), 0.025, places=4)
        self.assertAlmostEqual(ln.cdf().at(100.0), 0.975, places=4)
        lower, upper = ln.interval(0.95)
        self.assertAlmostEqual(lower, 10.0, places=4)
        self.assertAlmostEqual(upper, 100.0, places=4)

    def test_log_normal_from_quantiles(self):
        ln = LogNormal.from_quantiles(q1=0.05, v1=2.0, q2=0.95, v2=20.0)
        self.assertAlmostEqual(ln.cdf().at(2.0), 0.05, places=4)
        self.assertAlmostEqual(ln.cdf().at(20.0), 0.95, places=4)

    def test_log_normal_from_mean_and_std(self):
        ln = LogNormal.from_mean_and_std(mean=50.0, std=15.0)
        self.assertAlmostEqual(ln.mean(), 50.0, places=4)
        self.assertAlmostEqual(ln.std(), 15.0, places=4)

    def test_gamma_from_interval(self):
        # 95% interval between 1.0 and 15.0
        gam = Gamma.from_interval(1.0, 15.0, prob=0.95)
        self.assertAlmostEqual(gam.cdf().at(1.0), 0.025, places=4)
        self.assertAlmostEqual(gam.cdf().at(15.0), 0.975, places=4)
        lower, upper = gam.interval(0.95)
        self.assertAlmostEqual(lower, 1.0, places=4)
        self.assertAlmostEqual(upper, 15.0, places=4)

    def test_gamma_from_quantiles(self):
        gam = Gamma.from_quantiles(q1=0.10, v1=2.0, q2=0.90, v2=8.0)
        self.assertAlmostEqual(gam.cdf().at(2.0), 0.10, places=4)
        self.assertAlmostEqual(gam.cdf().at(8.0), 0.90, places=4)

    def test_gamma_from_mean_and_std(self):
        gam = Gamma.from_mean_and_std(mean=10.0, std=4.0)
        self.assertAlmostEqual(gam.mean(), 10.0, places=5)
        self.assertAlmostEqual(gam.std(), 4.0, places=5)

    def test_beta_from_interval(self):
        # 90% interval for a conversion rate between 0.05 and 0.25
        beta = Beta.from_interval(0.05, 0.25, prob=0.90)
        self.assertAlmostEqual(beta.cdf().at(0.05), 0.05, places=4)
        self.assertAlmostEqual(beta.cdf().at(0.25), 0.95, places=4)
        lower, upper = beta.interval(0.90)
        self.assertAlmostEqual(lower, 0.05, places=4)
        self.assertAlmostEqual(upper, 0.25, places=4)

    def test_beta_from_quantiles(self):
        beta = Beta.from_quantiles(q1=0.20, v1=0.30, q2=0.80, v2=0.70)
        self.assertAlmostEqual(beta.cdf().at(0.30), 0.20, places=4)
        self.assertAlmostEqual(beta.cdf().at(0.70), 0.80, places=4)

    def test_beta_from_mean_and_std(self):
        beta = Beta.from_mean_and_std(mean=0.4, std=0.1)
        self.assertAlmostEqual(beta.mean(), 0.4, places=5)
        self.assertAlmostEqual(beta.std(), 0.1, places=5)

    def test_beta_from_mean_and_std_invalid(self):
        # std too large: var >= mean * (1 - mean) = 0.4 * 0.6 = 0.24 -> std >= ~0.4899
        with self.assertRaises(ValueError):
            Beta.from_mean_and_std(mean=0.4, std=0.5)

    def test_inverse_gamma_from_interval(self):
        # 95% interval between 0.5 and 5.0
        ig = InverseGamma.from_interval(0.5, 5.0, prob=0.95)
        self.assertAlmostEqual(ig.cdf().at(0.5), 0.025, places=4)
        self.assertAlmostEqual(ig.cdf().at(5.0), 0.975, places=4)
        lower, upper = ig.interval(0.95)
        self.assertAlmostEqual(lower, 0.5, places=4)
        self.assertAlmostEqual(upper, 5.0, places=4)

    def test_inverse_gamma_from_quantiles(self):
        ig = InverseGamma.from_quantiles(q1=0.10, v1=1.0, q2=0.90, v2=4.0)
        self.assertAlmostEqual(ig.cdf().at(1.0), 0.10, places=4)
        self.assertAlmostEqual(ig.cdf().at(4.0), 0.90, places=4)

    def test_inverse_gamma_from_mean_and_std(self):
        ig = InverseGamma.from_mean_and_std(mean=3.0, std=1.5)
        self.assertAlmostEqual(ig.mean(), 3.0, places=4)
        self.assertAlmostEqual(ig.std(), 1.5, places=4)

    def test_continuous_uniform_from_interval(self):
        u = ContinuousUniform.from_interval(2.0, 8.0, prob=0.60)
        self.assertAlmostEqual(u.cdf().at(2.0), 0.20, places=5)
        self.assertAlmostEqual(u.cdf().at(8.0), 0.80, places=5)
        lower, upper = u.interval(0.60)
        self.assertAlmostEqual(lower, 2.0, places=5)
        self.assertAlmostEqual(upper, 8.0, places=5)

    def test_continuous_uniform_from_quantiles(self):
        u = ContinuousUniform.from_quantiles(q1=0.25, v1=10.0, q2=0.75, v2=20.0)
        self.assertAlmostEqual(u.a, 5.0, places=5)
        self.assertAlmostEqual(u.b, 25.0, places=5)

    def test_continuous_uniform_from_mean_and_std(self):
        u = ContinuousUniform.from_mean_and_std(mean=10.0, std=2.0)
        self.assertAlmostEqual(u.mean(), 10.0, places=5)
        self.assertAlmostEqual(u.std(), 2.0, places=5)

    def test_input_validation(self):
        # Invalid prob
        with self.assertRaises(ValueError):
            Normal.from_interval(1.0, 2.0, prob=1.5)
        with self.assertRaises(ValueError):
            Normal.from_interval(1.0, 2.0, prob=-0.1)

        # Inverted bounds
        with self.assertRaises(ValueError):
            Normal.from_interval(5.0, 2.0)

        # Negative bounds for positive-only distributions
        with self.assertRaises(ValueError):
            Gamma.from_interval(-1.0, 5.0)
        with self.assertRaises(ValueError):
            LogNormal.from_interval(-2.0, 3.0)
        with self.assertRaises(ValueError):
            Beta.from_interval(-0.1, 0.5)
        with self.assertRaises(ValueError):
            Beta.from_interval(0.1, 1.2)
