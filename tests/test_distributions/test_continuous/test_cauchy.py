from unittest.case import TestCase

from probability.distributions import Cauchy


class TestCauchy(TestCase):

    def setUp(self) -> None:
        self.c1 = Cauchy(x0=0.0, gamma=1.0)
        self.c2 = Cauchy(x0=5.0, gamma=2.0)

    def test_properties(self):
        self.assertEqual(self.c1.x0, 0.0)
        self.assertEqual(self.c1.gamma, 1.0)
        self.assertEqual(self.c1.loc, 0.0)
        self.assertEqual(self.c1.scale, 1.0)
        self.assertEqual(self.c1.mode(), 0.0)
        self.assertEqual(self.c1.median(), 0.0)

    def test_pdf_cdf_ppf(self):
        self.assertAlmostEqual(self.c1.cdf().at(0.0), 0.5, places=5)
        self.assertAlmostEqual(self.c1.ppf().at(0.5), 0.0, places=5)
        # HWHM: cdf at 1.0 is 0.75, at -1.0 is 0.25
        self.assertAlmostEqual(self.c1.cdf().at(1.0), 0.75, places=5)
        self.assertAlmostEqual(self.c1.cdf().at(-1.0), 0.25, places=5)

    def test_from_interval(self):
        c = Cauchy.from_interval(lower=-2.0, upper=2.0, prob=0.50)
        self.assertAlmostEqual(c.x0, 0.0, places=5)
        self.assertAlmostEqual(c.gamma, 2.0, places=5)
        lower, upper = c.interval(0.50)
        self.assertAlmostEqual(lower, -2.0, places=5)
        self.assertAlmostEqual(upper, 2.0, places=5)

    def test_from_quantiles(self):
        c = Cauchy.from_quantiles(q1=0.25, v1=3.0, q2=0.75, v2=7.0)
        self.assertAlmostEqual(c.x0, 5.0, places=5)
        self.assertAlmostEqual(c.gamma, 2.0, places=5)

    def test_fit(self):
        samples = self.c2.rvs(50_000, random_state=42)
        fitted = Cauchy.fit(samples)
        self.assertAlmostEqual(fitted.x0, 5.0, places=1)
        self.assertAlmostEqual(fitted.gamma, 2.0, places=1)

    def test_equality(self):
        c_same = Cauchy(x0=0.0, gamma=1.0)
        self.assertEqual(self.c1, c_same)
        self.assertNotEqual(self.c1, self.c2)

    def test_invalid_gamma(self):
        with self.assertRaises(ValueError):
            Cauchy(x0=0.0, gamma=-1.0)
        with self.assertRaises(ValueError):
            Cauchy(x0=0.0, gamma=0.0)
