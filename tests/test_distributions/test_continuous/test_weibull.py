from unittest.case import TestCase

from probability.distributions import Weibull


class TestWeibull(TestCase):

    def setUp(self) -> None:
        self.w1 = Weibull(k=1.0, lambda_=2.0)  # Exponential(rate=0.5)
        self.w2 = Weibull(k=2.0, lambda_=5.0)

    def test_properties(self):
        self.assertEqual(self.w1.k, 1.0)
        self.assertEqual(self.w1.lambda_, 2.0)
        self.assertEqual(self.w1.shape, 1.0)
        self.assertEqual(self.w1.scale, 2.0)
        self.assertEqual(self.w1.lower_bound, 0.0)

    def test_pdf_cdf_ppf(self):
        # When k=1, CDF is 1 - exp(-x/lambda) = 1 - exp(-2/2) = 1 - 1/e ~ 0.63212
        self.assertAlmostEqual(self.w1.cdf().at(2.0), 0.63212, places=4)
        self.assertAlmostEqual(self.w1.ppf().at(0.63212), 2.0, places=3)

    def test_mode(self):
        self.assertEqual(self.w1.mode(), 0.0)
        # For k=2, lambda=5, mode = 5 * (1/2)**(1/2) = 5 / sqrt(2) ~ 3.5355
        self.assertAlmostEqual(self.w2.mode(), 3.5355, places=4)

    def test_from_interval(self):
        w = Weibull.from_interval(lower=1.0, upper=10.0, prob=0.90)
        self.assertAlmostEqual(w.cdf().at(1.0), 0.05, places=4)
        self.assertAlmostEqual(w.cdf().at(10.0), 0.95, places=4)
        lower, upper = w.interval(0.90)
        self.assertAlmostEqual(lower, 1.0, places=4)
        self.assertAlmostEqual(upper, 10.0, places=4)

    def test_from_quantiles(self):
        w = Weibull.from_quantiles(q1=0.10, v1=2.0, q2=0.90, v2=8.0)
        self.assertAlmostEqual(w.cdf().at(2.0), 0.10, places=4)
        self.assertAlmostEqual(w.cdf().at(8.0), 0.90, places=4)

    def test_fit(self):
        samples = self.w2.rvs(50_000, random_state=42)
        fitted = Weibull.fit(samples)
        self.assertAlmostEqual(fitted.k, self.w2.k, places=1)
        self.assertAlmostEqual(fitted.lambda_, self.w2.lambda_, places=1)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            Weibull(k=-1.0, lambda_=1.0)
        with self.assertRaises(ValueError):
            Weibull(k=1.0, lambda_=0.0)
