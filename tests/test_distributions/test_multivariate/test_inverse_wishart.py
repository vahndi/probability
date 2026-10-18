from unittest import TestCase
import numpy as np
import pandas as pd
from probability.distributions.multivariate.inverse_wishart import InverseWishart
from probability.distributions.continuous.inverse_gamma import InverseGamma


class TestInverseWishart(TestCase):

    def setUp(self):
        self.psi = np.array([[3.0, 1.0], [1.0, 4.0]])
        self.dist = InverseWishart(nu=6.0, psi=self.psi, names=['x', 'y'])

    def test_properties(self):
        self.assertEqual(self.dist.nu, 6.0)
        self.assertEqual(self.dist.dim, 2)
        self.assertListEqual(self.dist.names, ['x', 'y'])
        self.assertAlmostEqual(self.dist.psi.loc['x', 'x'], 3.0)
        self.assertAlmostEqual(self.dist.psi.loc['x', 'y'], 1.0)

    def test_mean_and_mode(self):
        # mean = psi / (nu - d - 1) = psi / (6 - 2 - 1) = psi / 3
        mean = self.dist.mean()
        self.assertAlmostEqual(mean.loc['x', 'x'], 3.0 / 3.0)
        self.assertAlmostEqual(mean.loc['y', 'y'], 4.0 / 3.0)
        self.assertAlmostEqual(mean.loc['x', 'y'], 1.0 / 3.0)

        # mode = psi / (nu + d + 1) = psi / (6 + 2 + 1) = psi / 9
        mode = self.dist.mode()
        self.assertAlmostEqual(mode.loc['x', 'x'], 3.0 / 9.0)
        self.assertAlmostEqual(mode.loc['y', 'y'], 4.0 / 9.0)

    def test_marginal_variance(self):
        # Sigma_xx ~ InvGamma(alpha = (nu - d + 1)/2 = 5/2 = 2.5, beta = Psi_xx / 2 = 1.5)
        m_x = self.dist.marginal_variance('x')
        self.assertIsInstance(m_x, InverseGamma)
        self.assertAlmostEqual(m_x.alpha, 2.5)
        self.assertAlmostEqual(m_x.beta, 1.5)

        # Sigma_yy ~ InvGamma(alpha = 2.5, beta = 2.0)
        m_y = self.dist.marginal_variance(1)
        self.assertIsInstance(m_y, InverseGamma)
        self.assertAlmostEqual(m_y.alpha, 2.5)
        self.assertAlmostEqual(m_y.beta, 2.0)

    def test_rvs(self):
        # single draw
        draw = self.dist.rvs(num_samples=1, random_state=42)
        self.assertIsInstance(draw, pd.DataFrame)
        self.assertEqual(draw.shape, (2, 2))
        # positive definite check
        eigvals = np.linalg.eigvalsh(draw.values)
        self.assertTrue(np.all(eigvals > 0))

        # multiple draws
        draws = self.dist.rvs(num_samples=10, random_state=42)
        self.assertEqual(len(draws), 10)
        self.assertIsInstance(draws[0], pd.DataFrame)

    def test_pdf(self):
        x_mat = np.array([[1.0, 0.2], [0.2, 1.5]])
        p = self.dist.pdf(x_mat)
        self.assertGreater(p, 0.0)
