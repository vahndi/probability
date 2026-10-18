from unittest import TestCase
from matplotlib.axes import Axes
import numpy as np
import pandas as pd

from probability.distributions.conjugate.mv_normal_inverse_wishart_conjugate import (
    MVNormalInverseWishartConjugate
)
from probability.distributions.multivariate.mv_students_t import MVStudentsT
from probability.distributions.multivariate.inverse_wishart import InverseWishart
from probability.distributions.continuous.students_t import StudentsT
from probability.distributions.continuous.inverse_gamma import InverseGamma


class TestMVNormalInverseWishartConjugate(TestCase):

    def setUp(self):
        np.random.seed(42)
        self.n = 60
        self.true_mu = np.array([3.0, 7.0])
        self.true_cov = np.array([[2.0, 0.8], [0.8, 1.5]])
        raw_data = np.random.multivariate_normal(
            mean=self.true_mu,
            cov=self.true_cov,
            size=self.n
        )
        self.df = pd.DataFrame(raw_data, columns=['x1', 'x2'])

    def test_prior_initialization(self):
        conj = MVNormalInverseWishartConjugate(
            mu_0=[0.0, 0.0],
            kappa_0=1.0,
            nu_0=5.0,
            psi_0=[[1.0, 0.0], [0.0, 1.0]],
            names=['a', 'b']
        )
        self.assertEqual(conj.dim, 2)
        self.assertEqual(conj.n, 0)
        self.assertEqual(conj.kappa_0, 1.0)
        self.assertEqual(conj.nu_0, 5.0)

        # Prior distributions
        p_mu = conj.prior_mu()
        self.assertIsInstance(p_mu, MVStudentsT)
        self.assertEqual(p_mu.df, 5.0 - 2 + 1)  # 4.0

        p_sig = conj.prior_sigma()
        self.assertIsInstance(p_sig, InverseWishart)
        self.assertEqual(p_sig.df, 5.0)

    def test_posterior_updates(self):
        mu_0 = np.array([0.0, 0.0])
        kappa_0 = 2.0
        nu_0 = 4.0
        psi_0 = np.eye(2)

        conj = MVNormalInverseWishartConjugate(
            mu_0=mu_0,
            kappa_0=kappa_0,
            nu_0=nu_0,
            psi_0=psi_0,
            x=self.df
        )

        n = len(self.df)
        x_bar = self.df.mean().values
        # Analytical updates:
        expected_kappa_n = kappa_0 + n
        expected_nu_n = nu_0 + n
        expected_mu_n = (kappa_0 * mu_0 + n * x_bar) / expected_kappa_n

        self.assertEqual(conj.n, n)
        self.assertEqual(conj.kappa_n, expected_kappa_n)
        self.assertEqual(conj.nu_n, expected_nu_n)
        np.testing.assert_allclose(conj.mu_n.values, expected_mu_n, rtol=1e-5)

        # Posteriors
        post_mu = conj.posterior_mu()
        self.assertIsInstance(post_mu, MVStudentsT)
        self.assertEqual(post_mu.df, expected_nu_n - 2 + 1)

        post_sig = conj.posterior_sigma()
        self.assertIsInstance(post_sig, InverseWishart)
        self.assertEqual(post_sig.df, expected_nu_n)

        # Predictive
        pred = conj.posterior_predictive()
        self.assertIsInstance(pred, MVStudentsT)
        self.assertEqual(pred.df, expected_nu_n - 2 + 1)

    def test_univariate_marginals(self):
        conj = MVNormalInverseWishartConjugate.infer_posterior(self.df)

        # Marginal mu
        m_mu1 = conj.marginal_mu('x1')
        self.assertIsInstance(m_mu1, StudentsT)
        self.assertAlmostEqual(m_mu1.mu, self.df['x1'].mean(), places=1)

        m_mu2 = conj.marginal_mu(1)
        self.assertIsInstance(m_mu2, StudentsT)
        self.assertAlmostEqual(m_mu2.mu, self.df['x2'].mean(), places=1)

        # Marginal variance
        m_sig1 = conj.marginal_sigma_sq('x1')
        self.assertIsInstance(m_sig1, InverseGamma)
        # Expected value of marginal variance should be close to sample variance
        mean_var = m_sig1.beta / (m_sig1.alpha - 1.0)
        self.assertAlmostEqual(mean_var, self.df['x1'].var(), places=0)

    def test_sample_posterior(self):
        conj = MVNormalInverseWishartConjugate.infer_posterior(self.df)
        mu_samples, sig_samples = conj.sample_posterior(num_samples=100, random_state=42)

        self.assertIsInstance(mu_samples, pd.DataFrame)
        self.assertEqual(len(mu_samples), 100)
        self.assertEqual(len(sig_samples), 100)
        self.assertIsInstance(sig_samples[0], pd.DataFrame)

        # Mean of posterior draws for mu should be close to sample mean
        np.testing.assert_allclose(
            mu_samples.mean().values,
            self.df.mean().values,
            rtol=0.1
        )

    def test_plot(self):
        conj = MVNormalInverseWishartConjugate.infer_posterior(self.df)
        ax = conj.plot(dim1='x1', dim2='x2')
        self.assertIsInstance(ax, Axes)
