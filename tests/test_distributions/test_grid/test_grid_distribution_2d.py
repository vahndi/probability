import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from probability.distributions.grid import (
    GridDistribution1d,
    GridDistribution2d,
    BayesianGrid2d
)


class TestGridDistribution2d:

    def test_init_uniform(self):
        grid2d = GridDistribution2d.uniform(
            x_low=0.0, x_high=1.0, x_num=10,
            y_low=10.0, y_high=20.0, y_num=20,
            x_name='p', y_name='q'
        )
        assert grid2d.shape == (10, 20)
        assert grid2d.x_name == 'p'
        assert grid2d.y_name == 'q'
        assert np.isclose(grid2d.probs.sum(), 1.0)
        assert np.allclose(grid2d.probs, 1.0 / 200)

        # Means of uniform
        mu_x, mu_y = grid2d.mean()
        assert np.isclose(mu_x, 0.5, atol=0.01)
        assert np.isclose(mu_y, 15.0, atol=0.01)

    def test_validation_errors(self):
        with pytest.raises(ValueError, match="at least 2 points"):
            GridDistribution2d(x_grid=[1.0], y_grid=[1.0, 2.0])
        with pytest.raises(ValueError, match="at least 2 points"):
            GridDistribution2d(x_grid=[1.0, 2.0], y_grid=[1.0])
        with pytest.raises(ValueError, match="probs shape"):
            GridDistribution2d(
                x_grid=[1.0, 2.0],
                y_grid=[10.0, 20.0],
                probs=[[0.5, 0.5]]  # shape (1, 2) instead of (2, 2)
            )
        with pytest.raises(ValueError, match="non-negative"):
            GridDistribution2d(
                x_grid=[1.0, 2.0],
                y_grid=[10.0, 20.0],
                probs=[[0.5, -0.1], [0.3, 0.3]]
            )

    def test_from_independent(self):
        gx = GridDistribution1d.uniform(0.0, 10.0, 50, name='X')
        gy = GridDistribution1d.uniform(20.0, 40.0, 40, name='Y')
        g2d = GridDistribution2d.from_independent(gx, gy)

        assert g2d.shape == (50, 40)
        assert np.isclose(g2d.probs.sum(), 1.0)
        assert g2d.x_name == 'X'
        assert g2d.y_name == 'Y'

        # Independence implies zero correlation and covariance
        assert np.isclose(g2d.corr(), 0.0, atol=1e-10)
        cov = g2d.cov()
        assert np.isclose(cov[0, 1], 0.0, atol=1e-10)
        assert np.isclose(cov[1, 0], 0.0, atol=1e-10)

        # Marginals match original 1D distributions
        mx = g2d.marginal_x()
        my = g2d.marginal_y()
        assert np.isclose(mx.mean(), gx.mean())
        assert np.isclose(my.mean(), gy.mean())

    def test_from_function_and_mode(self):
        x = np.linspace(-3, 3, 61)
        y = np.linspace(-3, 3, 61)
        # Bivariate normal centered at (1.0, -1.0) with correlation
        fn = lambda xx, yy: np.exp(-0.5 * ((xx - 1.0)**2 + (yy + 1.0)**2 + 0.6 * (xx - 1.0) * (yy + 1.0)))

        g2d = GridDistribution2d.from_function(fn, x_grid=x, y_grid=y, x_name='x', y_name='y')
        assert np.isclose(g2d.probs.sum(), 1.0)

        # Mode should be exactly (1.0, -1.0)
        mx, my = g2d.mode()
        assert np.isclose(mx, 1.0, atol=0.1)
        assert np.isclose(my, -1.0, atol=0.1)

        # Mean should be close to (1.0, -1.0)
        mu_x, mu_y = g2d.mean()
        assert np.isclose(mu_x, 1.0, atol=0.1)
        assert np.isclose(mu_y, -1.0, atol=0.1)

    def test_conditionals(self):
        x = np.linspace(0, 10, 11)
        y = np.linspace(0, 10, 11)
        # Triangular joint distribution: P(X, Y) proportional to X * Y
        xx, yy = np.meshgrid(x, y, indexing='ij')
        probs = xx * yy
        probs = probs / probs.sum()

        g2d = GridDistribution2d(x_grid=x, y_grid=y, probs=probs)
        # Condition on Y = 5
        cond_x = g2d.conditional_x(given_y=5.0)
        assert isinstance(cond_x, GridDistribution1d)
        assert np.isclose(cond_x.probs.sum(), 1.0)
        # For P(X, Y) = X * Y, conditional P(X | Y=5) is proportional to X
        assert cond_x.probs[-1] > cond_x.probs[1]

        # Condition on X = 8
        cond_y = g2d.conditional_y(given_x=8.0)
        assert isinstance(cond_y, GridDistribution1d)
        assert np.isclose(cond_y.probs.sum(), 1.0)

    def test_sampling(self):
        g2d = GridDistribution2d.uniform(
            x_low=-2, x_high=2, x_num=41,
            y_low=10, y_high=20, y_num=51
        )
        samples = g2d.sample(size=500, seed=42)
        assert samples.shape == (500, 2)
        # Bounds check
        assert np.all(samples[:, 0] >= -2.0)
        assert np.all(samples[:, 0] <= 2.0)
        assert np.all(samples[:, 1] >= 10.0)
        assert np.all(samples[:, 1] <= 20.0)

        # Means approximately 0 and 15
        assert np.isclose(samples[:, 0].mean(), 0.0, atol=0.25)
        assert np.isclose(samples[:, 1].mean(), 15.0, atol=0.5)

    def test_hdr_mask(self):
        x = np.linspace(-3, 3, 31)
        y = np.linspace(-3, 3, 31)
        fn = lambda xx, yy: np.exp(-0.5 * (xx**2 + yy**2))
        g2d = GridDistribution2d.from_function(fn, x, y)

        mask_90 = g2d.hdr_mask(prob=0.90)
        assert mask_90.shape == g2d.shape
        # Total mass inside mask must be >= 0.90
        covered_prob = g2d.probs[mask_90].sum()
        assert covered_prob >= 0.90

        # Mode should be in the mask
        mx, my = g2d.mode()
        i = np.argmin(np.abs(x - mx))
        j = np.argmin(np.abs(y - my))
        assert mask_90[i, j]

    def test_bayesian_update_normal(self):
        # Infer unknown mu and sigma from Normal data
        prior = GridDistribution2d.uniform(
            x_low=8.0, x_high=12.0, x_num=50,
            y_low=1.0, y_high=4.0, y_num=50,
            x_name='mu', y_name='sigma'
        )

        np.random.seed(123)
        true_mu = 10.0
        true_sigma = 2.0
        data = np.random.normal(loc=true_mu, scale=true_sigma, size=60)

        posterior = prior.update_normal(data)
        assert np.isclose(posterior.probs.sum(), 1.0)

        post_mu, post_sigma = posterior.mode()
        # Should be close to sample mean and sample std
        sample_mean = float(np.mean(data))
        sample_std = float(np.std(data, ddof=1))
        assert np.isclose(post_mu, sample_mean, atol=0.15)
        assert np.isclose(post_sigma, sample_std, atol=0.20)

        # Marginal distributions
        post_mu_dist = posterior.marginal_x()
        assert np.isclose(post_mu_dist.mean(), sample_mean, atol=0.15)

    def test_bayesian_update_logistic_regression(self):
        # Infer intercept alpha and slope beta
        prior = GridDistribution2d.uniform(
            x_low=-2.0, x_high=2.0, x_num=40,
            y_low=0.0, y_high=3.0, y_num=40,
            x_name='alpha', y_name='beta'
        )

        # Synthetic logistic regression data with positive slope
        xs = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0] * 5)
        # Prob = 1 / (1 + exp(-(0.0 + 1.5 * x)))
        probs_true = 1.0 / (1.0 + np.exp(-(0.0 + 1.5 * xs)))
        ys = (probs_true > 0.5).astype(int)

        post = prior.update_logistic_regression(x_data=xs, y_binary=ys)
        assert np.isclose(post.probs.sum(), 1.0)

        # Beta should definitely be positive (mean > 0)
        post_alpha_mode, post_beta_mode = post.mode()
        assert post_beta_mode > 0.5

    def test_arbitrary_update_likelihood(self):
        g2d = GridDistribution2d.uniform(0, 1, 10, 0, 1, 10)
        # Likelihood is concentrated at upper right
        xx, yy = g2d.meshgrid
        lik = xx * yy
        post = g2d.update_likelihood(lik)
        assert np.isclose(post.probs.sum(), 1.0)
        mx, my = post.mode()
        assert mx == g2d.x_grid[-1]
        assert my == g2d.y_grid[-1]

    def test_plotting(self):
        g2d = GridDistribution2d.uniform(-2, 2, 20, -2, 2, 20)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax1 = g2d.plot_contour(ax=axes[0], hdr=0.90, title="Contour")
        ax2 = g2d.plot_heatmap(ax=axes[1], title="Heatmap")
        assert ax1 is axes[0]
        assert ax2 is axes[1]
        plt.close(fig)

    def test_alias(self):
        assert BayesianGrid2d is GridDistribution2d
