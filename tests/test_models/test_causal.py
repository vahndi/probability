import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from probability.distributions import StudentsT, Ratio, Boolean
from probability.models import IPW, AIPW, DoublyRobust


@pytest.fixture
def synthetic_causal_data():
    """
    Synthetic confounded dataset with known true ATE = 6.0:
        X1 ~ Normal(0, 1)
        X2 ~ Normal(0, 1)
        prob_T = sigmoid(0.6 * X1 - 0.4 * X2)
        T ~ Bernoulli(prob_T)
        Y = 20.0 + 6.0 * T + 3.0 * X1 - 2.0 * X2 + Normal(0, 1)
    """
    np.random.seed(42)
    n = 300
    x1 = np.random.normal(0, 1, size=n)
    x2 = np.random.normal(0, 1, size=n)
    z = 0.6 * x1 - 0.4 * x2
    p = 1.0 / (1.0 + np.exp(-z))
    t = (np.random.rand(n) < p).astype(float)
    y = 20.0 + 6.0 * t + 3.0 * x1 - 2.0 * x2 + np.random.normal(0, 1, size=n)

    return pd.DataFrame({'t': t, 'y': y, 'x1': x1, 'x2': x2})


class TestCausalEstimators:

    def test_ipw_ate_recovery(self, synthetic_causal_data):
        df = synthetic_causal_data
        ipw = IPW.fit(
            treatment=df['t'],
            outcome=df['y'],
            covariates=df[['x1', 'x2']],
            estimand='ATE'
        )

        assert isinstance(ipw, IPW)
        assert ipw.estimand == 'ATE'
        assert ipw.n_obs == 300
        # True ATE is 6.0
        assert np.isclose(ipw.estimate, 6.0, atol=0.8)
        assert ipw.std_err > 0.0

        # Effect distribution
        eff_dist = ipw.effect_distribution
        assert isinstance(eff_dist, StudentsT)
        assert eff_dist.nu == 299
        assert (eff_dist > 0) > 0.999

        # Summary
        summary = ipw.summary()
        assert 'estimate' in summary.columns
        assert 'std_err' in summary.columns
        assert '95% CI lower' in summary.columns
        assert '95% CI upper' in summary.columns
        assert summary.loc['ATE', '95% CI lower'] < 6.0 < summary.loc['ATE', '95% CI upper']

    def test_ipw_att(self, synthetic_causal_data):
        df = synthetic_causal_data
        ipw_att = IPW.fit(
            treatment=df['t'],
            outcome=df['y'],
            covariates=df[['x1', 'x2']],
            estimand='ATT'
        )
        assert ipw_att.estimand == 'ATT'
        assert np.isclose(ipw_att.estimate, 6.0, atol=0.8)

    def test_aipw_ate_recovery(self, synthetic_causal_data):
        df = synthetic_causal_data
        aipw = AIPW.fit(
            treatment=df['t'],
            outcome=df['y'],
            covariates=df[['x1', 'x2']],
            estimand='ATE'
        )

        assert isinstance(aipw, AIPW)
        assert aipw.estimand == 'ATE'
        assert aipw.n_obs == 300
        assert len(aipw.scores) == 300
        # Doubly robust estimate close to true ATE 6.0
        assert np.isclose(aipw.estimate, 6.0, atol=0.5)
        assert aipw.std_err > 0.0

        # Effect distribution
        eff_dist = aipw.effect_distribution
        assert isinstance(eff_dist, StudentsT)
        assert (eff_dist > 0) > 0.999

        # DoublyRobust alias
        dr = DoublyRobust.fit(
            treatment=df['t'],
            outcome=df['y'],
            covariates=df[['x1', 'x2']]
        )
        assert np.isclose(dr.estimate, aipw.estimate, atol=1e-10)

    def test_aipw_att(self, synthetic_causal_data):
        df = synthetic_causal_data
        aipw_att = AIPW.fit(
            treatment=df['t'],
            outcome=df['y'],
            covariates=df[['x1', 'x2']],
            estimand='ATT'
        )
        assert aipw_att.estimand == 'ATT'
        assert np.isclose(aipw_att.estimate, 6.0, atol=0.6)

    def test_bootstrapped_se(self, synthetic_causal_data):
        df = synthetic_causal_data
        ipw_boot = IPW.fit(
            treatment=df['t'],
            outcome=df['y'],
            covariates=df[['x1', 'x2']],
            n_bootstraps=25,
            seed=42
        )
        assert ipw_boot.std_err > 0.0

        aipw_boot = AIPW.fit(
            treatment=df['t'],
            outcome=df['y'],
            covariates=df[['x1', 'x2']],
            n_bootstraps=25,
            seed=42
        )
        assert aipw_boot.std_err > 0.0

    def test_love_plot_and_balance_summary(self, synthetic_causal_data):
        df = synthetic_causal_data
        ipw = IPW.fit(
            treatment=df['t'],
            outcome=df['y'],
            covariates=df[['x1', 'x2']]
        )

        bal = ipw.balance_summary()
        assert 'unweighted_diff' in bal.columns
        assert 'weighted_diff' in bal.columns
        assert 'abs_unweighted' in bal.columns
        assert 'abs_weighted' in bal.columns

        # Weighting should reduce standardized mean differences
        assert np.mean(bal['abs_weighted']) < np.mean(bal['abs_unweighted'])

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax1 = ipw.love_plot(ax=axes[0])
        ax2 = ipw.plot_propensity(ax=axes[1])
        assert ax1 is axes[0]
        assert ax2 is axes[1]
        plt.close(fig)

    def test_distribution_inputs(self, synthetic_causal_data):
        df = synthetic_causal_data
        t_dist = Boolean(df['t'])
        y_dist = Ratio(df['y'])

        aipw = AIPW.fit(
            treatment=t_dist,
            outcome=y_dist,
            covariates=df[['x1', 'x2']]
        )
        assert np.isclose(aipw.estimate, 6.0, atol=0.5)

    def test_error_handling(self):
        t_invalid = pd.Series([0, 1, 2, 0])
        y = pd.Series([1.0, 2.0, 3.0, 4.0])
        X = pd.DataFrame({'x': [1, 2, 3, 4]})

        with pytest.raises(ValueError, match="Treatment must be binary"):
            IPW.fit(treatment=t_invalid, outcome=y, covariates=X)

        t_single = pd.Series([1, 1, 1, 1])
        with pytest.raises(ValueError, match="Both treated.*and control"):
            IPW.fit(treatment=t_single, outcome=y, covariates=X)

        t_valid = pd.Series([0, 1, 0, 1])
        with pytest.raises(ValueError, match="Unknown estimand"):
            IPW.fit(treatment=t_valid, outcome=y, covariates=X, estimand='INVALID')
