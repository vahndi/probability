from unittest.case import TestCase
from unittest.mock import MagicMock, patch
import sys

from probability.distributions import (
    Normal, Beta, Gamma, InverseGamma, ContinuousUniform,
    LogNormal, Exponential, Cauchy, Weibull, Pareto,
    StudentsT, Laplace, Binomial, Poisson, NegativeBinomial,
    DiscreteUniform, Geometric
)


class TestPyMCBridge(TestCase):

    def test_missing_pymc_raises_importerror(self):
        # Ensure when pymc is not installed, calling .to_pymc() raises an informative ImportError
        n = Normal(0, 1)
        # Verify pymc is indeed not imported
        if 'pymc' in sys.modules:
            del sys.modules['pymc']

        with self.assertRaises(ImportError) as ctx:
            n.to_pymc('x')
        self.assertIn("pip install pymc", str(ctx.exception))

    def test_pymc_export_continuous_distributions(self):
        mock_pm = MagicMock()
        with patch.dict('sys.modules', {'pymc': mock_pm}):
            # Normal
            Normal(10, 2).to_pymc('mu_param')
            mock_pm.Normal.assert_called_with('mu_param', mu=10.0, sigma=2.0)

            # Beta
            Beta(2, 5).to_pymc('theta_param')
            mock_pm.Beta.assert_called_with('theta_param', alpha=2.0, beta=5.0)

            # Gamma (shape alpha, rate beta)
            Gamma(3, 0.5).to_pymc('lambda_param')
            mock_pm.Gamma.assert_called_with('lambda_param', alpha=3.0, beta=0.5)

            # InverseGamma
            InverseGamma(4, 2).to_pymc('sigma_sq')
            mock_pm.InverseGamma.assert_called_with('sigma_sq', alpha=4.0, beta=2.0)

            # ContinuousUniform
            ContinuousUniform(0, 100).to_pymc('prior_u')
            mock_pm.Uniform.assert_called_with('prior_u', lower=0.0, upper=100.0)

            # LogNormal
            LogNormal(1.5, 0.25).to_pymc('log_val')
            mock_pm.LogNormal.assert_called_with('log_val', mu=1.5, sigma=0.25)

            # Exponential
            Exponential(2.0).to_pymc('decay')
            mock_pm.Exponential.assert_called_with('decay', lam=2.0)

            # Cauchy
            Cauchy(0.0, 1.0).to_pymc('cauchy_prior')
            mock_pm.Cauchy.assert_called_with('cauchy_prior', alpha=0.0, beta=1.0)

            # Weibull
            Weibull(2.0, 3.0).to_pymc('weibull_prior')
            mock_pm.Weibull.assert_called_with('weibull_prior', alpha=2.0, beta=3.0)

            # Pareto
            Pareto(3.0, 1.0).to_pymc('pareto_prior')
            mock_pm.Pareto.assert_called_with('pareto_prior', alpha=3.0, m=1.0)

            # StudentsT
            StudentsT(nu=5, mu=0, sigma=1).to_pymc('robust_t')
            mock_pm.StudentT.assert_called_with('robust_t', nu=5.0, mu=0.0, sigma=1.0)

            # Laplace
            Laplace(mu=2.5, b=1.5).to_pymc('laplace_prior')
            mock_pm.Laplace.assert_called_with('laplace_prior', mu=2.5, b=1.5)

    def test_pymc_export_discrete_distributions(self):
        mock_pm = MagicMock()
        with patch.dict('sys.modules', {'pymc': mock_pm}):
            # Binomial
            Binomial(10, 0.3).to_pymc('k_obs', observed=3)
            mock_pm.Binomial.assert_called_with('k_obs', n=10, p=0.3, observed=3)

            # Geometric
            Geometric(0.4).to_pymc('geom_trials')
            mock_pm.Geometric.assert_called_with('geom_trials', p=0.4)

            # Poisson
            Poisson(4.0).to_pymc('counts')
            mock_pm.Poisson.assert_called_with('counts', mu=4.0)

            # NegativeBinomial
            NegativeBinomial(5, 0.5).to_pymc('neg_bin')
            mock_pm.NegativeBinomial.assert_called_with('neg_bin', n=5.0, p=0.5)

            # DiscreteUniform
            DiscreteUniform(1, 6).to_pymc('die_roll')
            mock_pm.DiscreteUniform.assert_called_with('die_roll', lower=1, upper=6)
