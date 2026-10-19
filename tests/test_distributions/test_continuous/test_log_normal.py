from math import exp
from unittest.case import TestCase

import numpy as np

from probability.distributions import LogNormal


class TestLogNormal(TestCase):

    def setUp(self) -> None:
        self.ln1 = LogNormal(mu=0.0, sigma=1.0)
        self.ln2 = LogNormal(mu=1.0, sigma=0.5)

    def test_properties(self):
        self.assertEqual(self.ln1.mu, 0.0)
        self.assertEqual(self.ln1.sigma, 1.0)
        self.assertAlmostEqual(self.ln1.mode(), exp(0.0 - 1.0 ** 2), places=6)
        self.assertAlmostEqual(self.ln1.median(), 1.0, places=6)

    def test_pdf_cdf(self):
        # Median of standard lognormal is 1.0
        self.assertAlmostEqual(self.ln1.cdf().at(1.0), 0.5, places=6)
        self.assertAlmostEqual(self.ln1.ppf().at(0.5), 1.0, places=6)

    def test_equality(self):
        ln_same = LogNormal(mu=0.0, sigma=1.0)
        self.assertEqual(self.ln1, ln_same)
        self.assertNotEqual(self.ln1, self.ln2)

    def test_fit(self):
        np.random.seed(42)
        samples = self.ln2.rvs(50_000, random_state=42)
        fitted = LogNormal.fit(samples)
        self.assertAlmostEqual(fitted.mu, self.ln2.mu, places=1)
        self.assertAlmostEqual(fitted.sigma, self.ln2.sigma, places=1)

    def test_invalid_sigma(self):
        with self.assertRaises(ValueError):
            LogNormal(mu=0.0, sigma=-1.0)
        with self.assertRaises(ValueError):
            LogNormal(mu=0.0, sigma=0.0)
