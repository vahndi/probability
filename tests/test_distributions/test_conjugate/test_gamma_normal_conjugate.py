from math import sqrt
from unittest.case import TestCase

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from pandas import Series, DataFrame

from probability.distributions import Gamma, StudentsT
from probability.distributions.conjugate import GammaNormalConjugate
from probability.distributions.conjugate.priors import VaguePrior


class TestGammaNormalConjugate(TestCase):

    def setUp(self) -> None:
        self.series = Series([1.5, 2.0, 2.5, 3.0, 1.0])
        self.mu = 2.0
        self.n_series = len(self.series)
        self.sum_sq = float(np.power(self.series - self.mu, 2).sum())

    def tearDown(self) -> None:
        plt.close('all')

    def test_posterior_parameters(self):
        conj = GammaNormalConjugate(
            alpha=2.0, beta=3.0, mu=self.mu, x=self.series
        )
        self.assertEqual(conj.n, self.n_series)
        self.assertAlmostEqual(conj.alpha_prime, 2.0 + self.n_series / 2.0)
        self.assertAlmostEqual(conj.beta_prime, 3.0 + self.sum_sq / 2.0)

        # Prior distribution
        prior = conj.prior()
        self.assertIsInstance(prior, Gamma)
        self.assertEqual(prior.alpha, 2.0)
        self.assertEqual(prior.beta, 3.0)

        # Posterior distribution
        post = conj.posterior()
        self.assertIsInstance(post, Gamma)
        self.assertAlmostEqual(post.alpha, 2.0 + self.n_series / 2.0)
        self.assertAlmostEqual(post.beta, 3.0 + self.sum_sq / 2.0)

    def test_predictive_distributions(self):
        conj = GammaNormalConjugate(
            alpha=2.0, beta=4.0, mu=self.mu, x=self.series
        )
        # Prior predictive
        prior_pred = conj.prior_predictive()
        self.assertIsInstance(prior_pred, StudentsT)
        self.assertEqual(prior_pred.nu, 4.0)
        self.assertEqual(prior_pred.mu, self.mu)
        self.assertAlmostEqual(prior_pred.sigma, sqrt(4.0 / 2.0))

        # Posterior predictive
        post_pred = conj.posterior_predictive()
        self.assertIsInstance(post_pred, StudentsT)
        self.assertAlmostEqual(post_pred.nu, 2.0 * conj.alpha_prime)
        self.assertEqual(post_pred.mu, self.mu)
        self.assertAlmostEqual(post_pred.sigma, sqrt(conj.beta_prime / conj.alpha_prime))

        # rvs
        samples = conj.rvs(100)
        self.assertEqual(len(samples), 100)

    def test_infer_posterior(self):
        expected = Gamma(
            alpha=VaguePrior.Gamma.alpha + self.n_series / 2.0,
            beta=VaguePrior.Gamma.beta + self.sum_sq / 2.0
        )
        actual = GammaNormalConjugate.infer_posterior(
            data=self.series, mu=self.mu
        )
        self.assertAlmostEqual(actual.alpha, expected.alpha, 10)
        self.assertAlmostEqual(actual.beta, expected.beta, 10)

    def test_infer_posteriors(self):
        df = DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'x': [1.0, 3.0, 2.0, 2.0]
        })
        res = GammaNormalConjugate.infer_posteriors(
            data=df, prob_vars='x', cond_vars='group', mu=2.0,
            alpha=1.0, beta=1.0
        )
        self.assertEqual(len(res), 2)
        self.assertIn('Gamma', res.columns)

    def test_plot(self):
        conj = GammaNormalConjugate(
            alpha=2.0, beta=3.0, mu=self.mu, x=self.series
        )
        fig = conj.plot()
        self.assertIsInstance(fig, Figure)
