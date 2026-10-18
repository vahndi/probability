from math import sqrt
from unittest.case import TestCase

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from pandas import Series

from probability.distributions import (
    InverseGamma, NormalInverseGammaConjugate, StudentsT
)


class TestNormalInverseGammaConjugate(TestCase):

    def setUp(self) -> None:
        self.series = Series([10.0, 12.0, 14.0, 16.0, 18.0])
        self.n = len(self.series)
        self.x_bar = float(self.series.mean())  # 14.0
        self.sum_sq_diff = float(np.power(self.series - self.x_bar, 2).sum())  # 40.0

    def tearDown(self) -> None:
        plt.close('all')

    def test_posterior_hyperparameters(self):
        mu_0, kappa_0, alpha_0, beta_0 = 10.0, 2.0, 3.0, 5.0
        conj = NormalInverseGammaConjugate(
            mu_0=mu_0, kappa_0=kappa_0, alpha_0=alpha_0, beta_0=beta_0,
            x=self.series
        )
        self.assertEqual(conj.n, self.n)
        self.assertEqual(conj.x_bar, self.x_bar)
        self.assertEqual(conj.sum_sq_diff, self.sum_sq_diff)

        # Expected updates
        expected_kappa_n = kappa_0 + self.n  # 7.0
        expected_mu_n = (kappa_0 * mu_0 + self.n * self.x_bar) / expected_kappa_n
        expected_alpha_n = alpha_0 + self.n / 2.0  # 5.5
        mean_penalty = (kappa_0 * self.n * (self.x_bar - mu_0) ** 2) / (2.0 * expected_kappa_n)
        expected_beta_n = beta_0 + self.sum_sq_diff / 2.0 + mean_penalty

        self.assertAlmostEqual(conj.kappa_n, expected_kappa_n)
        self.assertAlmostEqual(conj.mu_n, expected_mu_n)
        self.assertAlmostEqual(conj.alpha_n, expected_alpha_n)
        self.assertAlmostEqual(conj.beta_n, expected_beta_n)

    def test_marginal_distributions(self):
        mu_0, kappa_0, alpha_0, beta_0 = 0.0, 1.0, 2.0, 4.0
        conj = NormalInverseGammaConjugate(
            mu_0=mu_0, kappa_0=kappa_0, alpha_0=alpha_0, beta_0=beta_0,
            x=self.series
        )

        # Prior mu
        prior_mu = conj.prior_mu()
        self.assertIsInstance(prior_mu, StudentsT)
        self.assertEqual(prior_mu.nu, 2.0 * alpha_0)
        self.assertEqual(prior_mu.mu, mu_0)
        self.assertAlmostEqual(prior_mu.sigma, sqrt(beta_0 / (alpha_0 * kappa_0)))

        # Posterior mu
        post_mu = conj.posterior_mu()
        self.assertIsInstance(post_mu, StudentsT)
        self.assertEqual(post_mu.nu, 2.0 * conj.alpha_n)
        self.assertEqual(post_mu.mu, conj.mu_n)
        self.assertAlmostEqual(
            post_mu.sigma, sqrt(conj.beta_n / (conj.alpha_n * conj.kappa_n))
        )

        # Prior and posterior variance
        prior_sig = conj.prior_sigma_sq()
        self.assertIsInstance(prior_sig, InverseGamma)
        self.assertEqual(prior_sig.alpha, alpha_0)
        self.assertEqual(prior_sig.beta, beta_0)

        post_sig = conj.posterior_sigma_sq()
        self.assertIsInstance(post_sig, InverseGamma)
        self.assertEqual(post_sig.alpha, conj.alpha_n)
        self.assertEqual(post_sig.beta, conj.beta_n)

        # Generic prior and posterior accessors
        self.assertEqual(conj.prior('mu').mu, prior_mu.mu)
        self.assertEqual(conj.posterior('sigma_sq').alpha, post_sig.alpha)

    def test_predictive_distributions(self):
        conj = NormalInverseGammaConjugate(
            mu_0=10.0, kappa_0=1.0, alpha_0=2.0, beta_0=4.0,
            x=self.series
        )
        prior_pred = conj.prior_predictive()
        self.assertIsInstance(prior_pred, StudentsT)
        self.assertEqual(prior_pred.nu, 4.0)

        post_pred = conj.posterior_predictive()
        self.assertIsInstance(post_pred, StudentsT)
        self.assertEqual(post_pred.nu, 2.0 * conj.alpha_n)
        self.assertEqual(post_pred.mu, conj.mu_n)

        samples = conj.rvs(100)
        self.assertEqual(len(samples), 100)

    def test_sample_posterior(self):
        conj = NormalInverseGammaConjugate(
            mu_0=10.0, kappa_0=1.0, alpha_0=2.0, beta_0=4.0,
            x=self.series
        )
        mu_samples, sig_samples = conj.sample_posterior(num_samples=500)
        self.assertEqual(len(mu_samples), 500)
        self.assertEqual(len(sig_samples), 500)
        self.assertTrue((sig_samples > 0).all())

    def test_infer_posterior(self):
        conj = NormalInverseGammaConjugate.infer_posterior(self.series)
        self.assertIsInstance(conj, NormalInverseGammaConjugate)
        self.assertEqual(conj.n, self.n)
        post_mu = conj.posterior_mu()
        self.assertIsInstance(post_mu, StudentsT)
        self.assertAlmostEqual(post_mu.mu, self.x_bar, places=1)

    def test_plot(self):
        conj = NormalInverseGammaConjugate(
            mu_0=10.0, kappa_0=1.0, alpha_0=2.0, beta_0=4.0,
            x=self.series
        )
        fig = conj.plot()
        self.assertIsInstance(fig, Figure)
