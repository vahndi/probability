import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from probability.models import (
    LaplaceApproximation,
    QuadraticApproximation,
    Quap
)
from probability.distributions import Normal, MVNormal


class TestLaplaceApproximation:

    def test_1d_binomial_proportion(self):
        # 140 heads in 250 spins (The Euro problem)
        k, n = 140, 250

        # Log-posterior: log prior (flat on (0, 1)) + log binomial likelihood
        def log_post(theta):
            p = theta[0]
            if p <= 0.0 or p >= 1.0:
                return -np.inf
            # k * log(p) + (n - k) * log(1 - p)
            return k * np.log(p) + (n - k) * np.log(1.0 - p)

        laplace = LaplaceApproximation.fit(
            log_posterior_fn=log_post,
            start_params={'p': 0.5},
            bounds={'p': (0.01, 0.99)}
        )

        assert laplace.success
        assert laplace.ndim == 1
        assert laplace.param_names == ['p']

        # MAP estimate matches sample proportion k/n
        expected_p = k / n
        assert np.isclose(laplace.mode['p'], expected_p, atol=1e-3)

        # Standard error: sqrt(p * (1 - p) / n)
        expected_se = np.sqrt(expected_p * (1.0 - expected_p) / n)
        assert np.isclose(laplace.std['p'], expected_se, atol=1e-3)

        # 1D marginal
        marg = laplace.marginal('p')
        assert isinstance(marg, Normal)
        assert np.isclose(marg.mu, expected_p, atol=1e-3)
        assert np.isclose(marg.sigma, expected_se, atol=1e-3)

        # Summary table
        summ = laplace.summary(hdi=0.95)
        assert 'mean' in summ.columns
        assert 'std' in summ.columns
        assert '95% HDI lower' in summ.columns
        assert '95% HDI upper' in summ.columns
        assert summ.loc['p', '95% HDI lower'] < expected_p < summ.loc['p', '95% HDI upper']

    def test_2d_normal_parameters(self):
        # Infer unknown mu and sigma from Normal data
        np.random.seed(42)
        true_mu, true_sigma = 175.0, 8.0
        n_obs = 100
        data = np.random.normal(loc=true_mu, scale=true_sigma, size=n_obs)
        sample_mean = float(np.mean(data))
        sample_std = float(np.std(data, ddof=0))  # MLE denominator N

        def log_post(theta):
            mu, sigma = theta[0], theta[1]
            if sigma <= 0.0:
                return -np.inf
            # Log likelihood: -n * log(sigma) - sum((d - mu)^2) / (2 * sigma^2)
            ss = np.sum((data - mu) ** 2)
            return -n_obs * np.log(sigma) - ss / (2.0 * sigma ** 2)

        laplace = LaplaceApproximation.fit(
            log_posterior_fn=log_post,
            start_params=[170.0, 5.0],
            param_names=['mu', 'sigma'],
            bounds=[(150.0, 200.0), (1.0, 20.0)]
        )

        assert laplace.success
        assert laplace.ndim == 2

        # MAP estimates match sample mean and std
        assert np.isclose(laplace.mode['mu'], sample_mean, atol=0.05)
        assert np.isclose(laplace.mode['sigma'], sample_std, atol=0.05)

        # Theoretical standard errors: SE(mu) = sigma / sqrt(n), SE(sigma) = sigma / sqrt(2n)
        se_mu_theory = true_sigma / np.sqrt(n_obs)
        se_sigma_theory = true_sigma / np.sqrt(2.0 * n_obs)
        assert np.isclose(laplace.std['mu'], se_mu_theory, atol=0.15)
        assert np.isclose(laplace.std['sigma'], se_sigma_theory, atol=0.15)

        # In a normal distribution, mu and sigma are asymptotically orthogonal (cov = 0)
        cov_matrix = laplace.cov
        assert np.isclose(cov_matrix.loc['mu', 'sigma'], 0.0, atol=0.1)

        # Joint posterior distribution
        post_dist = laplace.posterior
        assert isinstance(post_dist, MVNormal)

    def test_fit_model_with_separate_prior_and_likelihood(self):
        # Linear model: y = alpha + beta * x + eps
        np.random.seed(123)
        x = np.linspace(-2, 2, 40)
        true_alpha, true_beta, true_sigma = 2.0, 1.5, 0.5
        y = true_alpha + true_beta * x + np.random.normal(0, true_sigma, size=len(x))

        # Priors: alpha ~ N(0, 10^2), beta ~ N(0, 10^2), sigma ~ HalfNormal
        def log_prior(theta):
            alpha, beta, sigma = theta
            if sigma <= 0:
                return -np.inf
            return (
                -0.5 * (alpha / 10.0) ** 2
                - 0.5 * (beta / 10.0) ** 2
                - 0.5 * (sigma / 2.0) ** 2
            )

        def log_likelihood(theta, d):
            alpha, beta, sigma = theta
            xs, ys = d['x'], d['y']
            resid = ys - (alpha + beta * xs)
            return -len(xs) * np.log(sigma) - np.sum(resid ** 2) / (2.0 * sigma ** 2)

        data = {'x': x, 'y': y}
        laplace = LaplaceApproximation.fit_model(
            log_prior_fn=log_prior,
            log_likelihood_fn=log_likelihood,
            start_params={'alpha': 0.0, 'beta': 0.0, 'sigma': 1.0},
            data=data,
            bounds={'alpha': (-20, 20), 'beta': (-20, 20), 'sigma': (0.01, 10)}
        )

        assert laplace.success
        assert np.isclose(laplace.mode['alpha'], true_alpha, atol=0.25)
        assert np.isclose(laplace.mode['beta'], true_beta, atol=0.25)
        assert np.isclose(laplace.mode['sigma'], true_sigma, atol=0.20)

    def test_sampling_and_plotting(self):
        laplace = LaplaceApproximation(
            mode=[5.0, 10.0],
            cov=[[1.0, 0.2], [0.2, 2.0]],
            param_names=['x', 'y']
        )
        samples = laplace.sample(size=300, seed=42)
        assert samples.shape == (300, 2)
        assert list(samples.columns) == ['x', 'y']
        assert np.isclose(samples['x'].mean(), 5.0, atol=0.2)
        assert np.isclose(samples['y'].mean(), 10.0, atol=0.3)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax_forest = laplace.plot_forest(ax=axes[0])
        ax_marg = laplace.plot_marginal('x', ax=axes[1])
        assert ax_forest is axes[0]
        assert ax_marg is axes[1]
        plt.close(fig)

    def test_aliases(self):
        assert QuadraticApproximation is LaplaceApproximation
        assert Quap is LaplaceApproximation
