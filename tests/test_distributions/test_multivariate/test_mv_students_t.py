from unittest import TestCase
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from probability.distributions.multivariate.mv_students_t import MVStudentsT
from probability.distributions.continuous.students_t import StudentsT


class TestMVStudentsT(TestCase):

    def setUp(self):
        self.mu = [1.0, 2.0]
        self.sigma = [[2.0, 0.5], [0.5, 3.0]]
        self.dist = MVStudentsT(nu=5.0, mu=self.mu, sigma=self.sigma, names=['a', 'b'])

    def test_properties(self):
        self.assertEqual(self.dist.nu, 5.0)
        self.assertEqual(self.dist.num_dims, 2)
        self.assertListEqual(self.dist.names, ['a', 'b'])
        self.assertAlmostEqual(self.dist.mu['a'], 1.0)
        self.assertAlmostEqual(self.dist.mu['b'], 2.0)
        self.assertAlmostEqual(self.dist.sigma.loc['a', 'b'], 0.5)

    def test_mean_and_cov(self):
        # Mean for nu=5 > 1 is mu
        mean = self.dist.mean()
        self.assertAlmostEqual(mean['a'], 1.0)
        self.assertAlmostEqual(mean['b'], 2.0)

        # Cov for nu=5 > 2 is sigma * (5 / 3)
        cov = self.dist.cov()
        self.assertAlmostEqual(cov.loc['a', 'a'], 2.0 * 5.0 / 3.0)
        self.assertAlmostEqual(cov.loc['b', 'b'], 3.0 * 5.0 / 3.0)
        self.assertAlmostEqual(cov.loc['a', 'b'], 0.5 * 5.0 / 3.0)

    def test_marginal(self):
        m_a = self.dist.marginal('a')
        self.assertIsInstance(m_a, StudentsT)
        self.assertAlmostEqual(m_a.nu, 5.0)
        self.assertAlmostEqual(m_a.mu, 1.0)
        self.assertAlmostEqual(m_a.sigma, np.sqrt(2.0))

        m_b = self.dist.marginal(1)
        self.assertIsInstance(m_b, StudentsT)
        self.assertAlmostEqual(m_b.nu, 5.0)
        self.assertAlmostEqual(m_b.mu, 2.0)
        self.assertAlmostEqual(m_b.sigma, np.sqrt(3.0))

    def test_rvs(self):
        samples = self.dist.rvs(num_samples=100, random_state=42)
        self.assertIsInstance(samples, pd.DataFrame)
        self.assertEqual(len(samples), 100)
        self.assertListEqual(list(samples.columns), ['a', 'b'])
        # Sample mean should be close to mu
        self.assertAlmostEqual(samples['a'].mean(), 1.0, places=0)
        self.assertAlmostEqual(samples['b'].mean(), 2.0, places=0)

    def test_pdf(self):
        val = self.dist.pdf().at([1.0, 2.0])
        self.assertGreater(val, 0.0)

    def test_plot_2d(self):
        x1 = np.linspace(-3, 5, 20)
        x2 = np.linspace(-2, 6, 20)
        ax = self.dist.plot_2d(x1=x1, x2=x2)
        self.assertIsInstance(ax, Axes)
