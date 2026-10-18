from unittest.case import TestCase

from pandas import Series

from probability.distributions import (
    Beta,
    BetaNegativeBinomialConjugate,
    NegativeBinomial,
)


class TestBetaNegativeBinomialConjugate(TestCase):

    def setUp(self) -> None:
        # 1 experiment: r=5 successes, observed k=10 failures
        # Prior Beta(alpha=2, beta=3)
        self.conj = BetaNegativeBinomialConjugate(
            r=5, k=10, n=1,
            alpha=2.0, beta=3.0
        )

    def test_prior(self):
        prior = self.conj.prior()
        self.assertEqual(prior, Beta(alpha=2.0, beta=3.0))

    def test_posterior(self):
        # alpha' = 2 + 10 = 12
        # beta' = 3 + 1 * 5 = 8
        post = self.conj.posterior()
        self.assertEqual(post, Beta(alpha=12.0, beta=8.0))

    def test_likelihood(self):
        lik = self.conj.likelihood()
        self.assertIsInstance(lik, NegativeBinomial)
        self.assertEqual(lik.r, 5)
        self.assertAlmostEqual(lik.p, 10.0 / 15.0, places=5)

    def test_infer_posterior(self):
        # Observed failures across 3 experiments: [4, 6, 2]
        # Target r=3 per experiment
        # k_total = 12, n=3
        # Prior is Uniform(1, 1)
        # Expected posterior: Beta(1 + 12, 1 + 3*3) = Beta(13, 10)
        data = Series([4, 6, 2])
        post = BetaNegativeBinomialConjugate.infer_posterior(data=data, r=3)
        self.assertEqual(post, Beta(alpha=13.0, beta=10.0))

    def test_negative_binomial_from_mean_and_variance(self):
        # Mean = 6.0, Variance = 18.0 (overdispersed: 18 > 6)
        # p = 1 - 6/18 = 2/3
        # r = 36 / (18 - 6) = 36 / 12 = 3.0
        nb = NegativeBinomial.from_mean_and_variance(mean=6.0, variance=18.0)
        self.assertAlmostEqual(nb.r, 3.0, places=5)
        self.assertAlmostEqual(nb.p, 2.0 / 3.0, places=5)
        self.assertAlmostEqual(nb.mean(), 6.0, places=5)
        self.assertAlmostEqual(nb.var(), 18.0, places=4)

    def test_negative_binomial_fit(self):
        data = [2, 4, 3, 5, 8, 2, 6, 10]
        nb = NegativeBinomial.fit(data)
        self.assertIsInstance(nb, NegativeBinomial)
        self.assertGreater(nb.r, 0)
        self.assertGreater(nb.p, 0)
        self.assertLess(nb.p, 1)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            BetaNegativeBinomialConjugate(r=0, k=5)
        with self.assertRaises(ValueError):
            BetaNegativeBinomialConjugate(r=5, k=-1)
        with self.assertRaises(ValueError):
            NegativeBinomial.from_mean_and_variance(mean=10.0, variance=5.0)
