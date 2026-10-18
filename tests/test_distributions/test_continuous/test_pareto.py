from unittest.case import TestCase

from probability.distributions import Pareto


class TestPareto(TestCase):

    def setUp(self) -> None:
        self.p1 = Pareto(alpha=2.5, x_m=1.0)
        self.p2 = Pareto(alpha=3.0, x_m=10.0)

    def test_properties(self):
        self.assertEqual(self.p1.alpha, 2.5)
        self.assertEqual(self.p1.x_m, 1.0)
        self.assertEqual(self.p1.shape, 2.5)
        self.assertEqual(self.p1.scale, 1.0)
        self.assertEqual(self.p1.lower_bound, 1.0)
        self.assertEqual(self.p1.mode(), 1.0)

    def test_pdf_cdf_ppf(self):
        # CDF(x) = 1 - (x_m / x)^alpha
        # CDF(2) = 1 - (1/2)^2.5 = 1 - 0.17677 = 0.82322
        self.assertAlmostEqual(self.p1.cdf().at(2.0), 0.82322, places=4)
        self.assertAlmostEqual(self.p1.ppf().at(0.82322), 2.0, places=3)

    def test_from_interval(self):
        p = Pareto.from_interval(lower=12.0, upper=50.0, prob=0.90)
        self.assertAlmostEqual(p.cdf().at(12.0), 0.05, places=4)
        self.assertAlmostEqual(p.cdf().at(50.0), 0.95, places=4)
        lower, upper = p.interval(0.90)
        self.assertAlmostEqual(lower, 12.0, places=4)
        self.assertAlmostEqual(upper, 50.0, places=4)

    def test_from_quantiles(self):
        p = Pareto.from_quantiles(q1=0.20, v1=15.0, q2=0.80, v2=60.0)
        self.assertAlmostEqual(p.cdf().at(15.0), 0.20, places=4)
        self.assertAlmostEqual(p.cdf().at(60.0), 0.80, places=4)

    def test_fit(self):
        samples = self.p2.rvs(50_000, random_state=42)
        fitted = Pareto.fit(samples)
        self.assertAlmostEqual(fitted.alpha, self.p2.alpha, places=1)
        self.assertAlmostEqual(fitted.x_m, self.p2.x_m, places=1)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            Pareto(alpha=-1.0, x_m=1.0)
        with self.assertRaises(ValueError):
            Pareto(alpha=1.0, x_m=0.0)
