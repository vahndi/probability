from unittest.case import TestCase

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from pandas import DataFrame

from probability.distributions import InverseGamma, StudentsT
from probability.models import BayesianLinearRegression


class TestBayesianLinearRegression(TestCase):

    def setUp(self) -> None:
        np.random.seed(42)
        n = 100
        self.x1 = np.random.normal(0, 1, n)
        self.x2 = np.random.normal(2, 1, n)
        # y = 3.0 + 2.0 * x1 - 1.5 * x2 + noise(sigma=0.5)
        self.true_intercept = 3.0
        self.true_beta1 = 2.0
        self.true_beta2 = -1.5
        self.noise = np.random.normal(0, 0.5, n)
        self.y = self.true_intercept + self.true_beta1 * self.x1 + self.true_beta2 * self.x2 + self.noise

        self.df = DataFrame({
            'x1': self.x1,
            'x2': self.x2,
            'y': self.y
        })

    def tearDown(self) -> None:
        plt.close('all')

    def test_fit_and_coefficients(self):
        model = BayesianLinearRegression.fit(
            X=self.df[['x1', 'x2']],
            y=self.df['y'],
            fit_intercept=True
        )

        self.assertEqual(model.n_obs, 100)
        self.assertEqual(model.feature_names, ['intercept', 'x1', 'x2'])

        # Check posterior coefficients
        beta_intercept = model.coef('intercept')
        beta_x1 = model.coef('x1')
        beta_x2 = model.coef('x2')

        self.assertIsInstance(beta_intercept, StudentsT)
        self.assertIsInstance(beta_x1, StudentsT)
        self.assertIsInstance(beta_x2, StudentsT)

        self.assertAlmostEqual(beta_intercept.mu, self.true_intercept, delta=0.2)
        self.assertAlmostEqual(beta_x1.mu, self.true_beta1, delta=0.2)
        self.assertAlmostEqual(beta_x2.mu, self.true_beta2, delta=0.2)

        # Residual variance
        sig_sq = model.sigma_sq
        self.assertIsInstance(sig_sq, InverseGamma)
        # True variance = 0.5^2 = 0.25
        sig_sq_mode = sig_sq.beta / (sig_sq.alpha + 1)
        self.assertAlmostEqual(sig_sq_mode, 0.25, delta=0.1)

    def test_probabilities_and_treatment_effect(self):
        model = BayesianLinearRegression.fit(
            X=self.df[['x1', 'x2']],
            y=self.df['y'],
            fit_intercept=True
        )

        # x1 has true effect +2.0, so P(beta_x1 > 0) should be near 1.0
        self.assertGreater(model.prob_positive('x1'), 0.999)
        self.assertLess(model.prob_negative('x1'), 0.001)

        # x2 has true effect -1.5, so P(beta_x2 < 0) should be near 1.0
        self.assertGreater(model.prob_negative('x2'), 0.999)

        # Treatment effect
        te = model.treatment_effect('x1')
        self.assertIsInstance(te, StudentsT)
        self.assertAlmostEqual(te.mu, model.coef('x1').mu)

    def test_predictions(self):
        model = BayesianLinearRegression.fit(
            X=self.df[['x1', 'x2']],
            y=self.df['y'],
            fit_intercept=True
        )

        X_new = DataFrame({
            'x1': [0.0, 1.0],
            'x2': [2.0, 2.0]
        })

        # 1. Point prediction
        preds = model.predict(X_new)
        self.assertEqual(len(preds), 2)
        # At (0, 2): y = 3 + 2(0) - 1.5(2) = 0.0
        self.assertAlmostEqual(preds.iloc[0], 0.0, delta=0.3)
        # At (1, 2): y = 3 + 2(1) - 1.5(2) = 2.0
        self.assertAlmostEqual(preds.iloc[1], 2.0, delta=0.3)

        # 2. Predictive distributions
        dists = model.predict_dist(X_new)
        self.assertEqual(len(dists), 2)
        self.assertIsInstance(dists.iloc[0], StudentsT)
        self.assertAlmostEqual(dists.iloc[0].mu, preds.iloc[0])

        # 3. Intervals
        intervals = model.predict_interval(X_new, hdi=0.95)
        self.assertEqual(len(intervals), 2)
        self.assertIn('mean', intervals.columns)
        self.assertIn('hdi_95_lower', intervals.columns)
        self.assertIn('hdi_95_upper', intervals.columns)
        self.assertTrue((intervals['hdi_95_lower'] < intervals['mean']).all())
        self.assertTrue((intervals['hdi_95_upper'] > intervals['mean']).all())

    def test_sample_posterior(self):
        model = BayesianLinearRegression.fit(
            X=self.df[['x1', 'x2']],
            y=self.df['y'],
            fit_intercept=True
        )
        beta_samples, sig_sq_samples = model.sample_posterior(num_samples=200)
        self.assertEqual(len(beta_samples), 200)
        self.assertEqual(len(sig_sq_samples), 200)
        self.assertIn('x1', beta_samples.columns)
        self.assertIn('intercept', beta_samples.columns)
        self.assertTrue((sig_sq_samples > 0).all())

    def test_summary_and_plots(self):
        model = BayesianLinearRegression.fit(
            X=self.df[['x1']],
            y=self.df['y'],
            fit_intercept=True
        )
        # Summary
        s = model.summary()
        self.assertIn('beta[x1]', s.index)
        self.assertIn('sigma_sq', s.index)
        self.assertIn('P(>0)', s.columns)

        # Plot coefficients
        ax_coef = model.plot_coefficients()
        self.assertIsInstance(ax_coef, Axes)

        # Plot fit
        ax_fit = model.plot_fit(x_var='x1')
        self.assertIsInstance(ax_fit, Axes)
