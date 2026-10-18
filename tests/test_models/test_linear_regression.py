import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from probability.distributions import Ratio, StudentsT
from probability.models import LinearRegression, OLS


@pytest.fixture
def synthetic_ols_data():
    """
    Synthetic regression data with true parameters:
        y = 5.0 + 2.5 * x1 - 1.8 * x2 + eps
    """
    np.random.seed(42)
    n = 200
    x1 = np.random.normal(loc=10.0, scale=2.0, size=n)
    x2 = np.random.normal(loc=5.0, scale=1.5, size=n)
    eps = np.random.normal(loc=0.0, scale=1.0, size=n)
    y = 5.0 + 2.5 * x1 - 1.8 * x2 + eps

    df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
    return df


class TestLinearRegression:

    def test_ols_point_estimates_and_properties(self, synthetic_ols_data):
        df = synthetic_ols_data
        model = LinearRegression.fit(X=df[['x1', 'x2']], y=df['y'], cov_type='nonrobust')

        assert isinstance(model, LinearRegression)
        assert len(model.feature_names) == 3
        assert model.feature_names == ['intercept', 'x1', 'x2']
        assert model.n_obs == 200
        assert model.df_resid == 197
        assert model.df_model == 2

        # Point estimates close to true parameters
        assert np.isclose(model.params['intercept'], 5.0, atol=1.0)
        assert np.isclose(model.params['x1'], 2.5, atol=0.1)
        assert np.isclose(model.params['x2'], -1.8, atol=0.1)

        # R^2 and F-stat
        assert 0.85 < model.rsquared < 1.0
        assert model.rsquared_adj <= model.rsquared
        assert model.f_statistic > 100.0
        assert model.f_pvalue < 1e-10

        # Residuals and fitted values
        assert len(model.residuals) == 200
        assert len(model.fitted_values) == 200
        assert np.isclose(np.mean(model.residuals), 0.0, atol=1e-10)

    def test_ols_alias(self, synthetic_ols_data):
        df = synthetic_ols_data
        model1 = LinearRegression.fit(X=df[['x1']], y=df['y'])
        model2 = OLS.fit(X=df[['x1']], y=df['y'])
        assert np.allclose(model1.params.values, model2.params.values)

    def test_robust_covariances(self, synthetic_ols_data):
        df = synthetic_ols_data
        for c_type in ['nonrobust', 'HC0', 'HC1', 'HC2', 'HC3']:
            m = LinearRegression.fit(X=df[['x1', 'x2']], y=df['y'], cov_type=c_type)
            assert m.cov_type == (c_type.upper() if c_type.startswith('hc') else c_type)
            assert np.all(m.bse.values > 0.0)

        # HC1 is exactly n / (n - k) * HC0
        m_hc0 = LinearRegression.fit(X=df[['x1', 'x2']], y=df['y'], cov_type='HC0')
        m_hc1 = LinearRegression.fit(X=df[['x1', 'x2']], y=df['y'], cov_type='HC1')
        ratio = float(m.n_obs) / float(m.df_resid)
        assert np.allclose(m_hc1.cov_params.values, ratio * m_hc0.cov_params.values)

    def test_clustered_standard_errors(self):
        np.random.seed(42)
        n = 150
        n_clusters = 10
        clusters = np.repeat(np.arange(n_clusters), n // n_clusters)
        x = np.random.normal(0, 1, n)
        # Add cluster random effect
        cluster_effects = np.random.normal(0, 2, n_clusters)
        y = 3.0 + 2.0 * x + cluster_effects[clusters] + np.random.normal(0, 0.5, n)

        m_cluster = LinearRegression.fit(
            X=pd.Series(x, name='x'),
            y=pd.Series(y, name='y'),
            cluster=clusters
        )

        assert m_cluster.cov_type == 'cluster'
        assert m_cluster.df_resid == n_clusters - 1  # 9 degrees of freedom
        assert np.all(m_cluster.bse.values > 0.0)

        # Error if only 1 cluster
        with pytest.raises(ValueError, match="at least 2 distinct clusters"):
            LinearRegression.fit(X=pd.Series(x, name='x'), y=pd.Series(y, name='y'), cluster=np.zeros(n))

    def test_students_t_coefficient_distributions(self, synthetic_ols_data):
        df = synthetic_ols_data
        model = LinearRegression.fit(X=df[['x1', 'x2']], y=df['y'], cov_type='HC3')

        t_x1 = model.coefficient('x1')
        assert isinstance(t_x1, StudentsT)
        assert t_x1.nu == model.df_resid
        assert np.isclose(t_x1.mu, model.params['x1'])
        assert np.isclose(t_x1.sigma, model.bse['x1'])

        # Prob(x1 > 0) should be practically 1.0 since beta_1 = 2.5
        prob_pos = t_x1 > 0
        assert prob_pos > 0.999

        # Dictionary of all coefficients
        all_coefs = model.coefficients
        assert set(all_coefs.keys()) == {'intercept', 'x1', 'x2'}

        # Confidence interval in summary matches Student's t interval
        summary = model.summary(alpha=0.05)
        low_t, high_t = t_x1.interval(0.95)
        assert np.isclose(summary.loc['x1', '95% CI lower'], low_t, atol=1e-4)
        assert np.isclose(summary.loc['x1', '95% CI upper'], high_t, atol=1e-4)

    def test_predict_and_intervals(self, synthetic_ols_data):
        df = synthetic_ols_data
        model = LinearRegression.fit(X=df[['x1', 'x2']], y=df['y'])

        new_data = pd.DataFrame({'x1': [10.0, 12.0], 'x2': [5.0, 4.0]})
        preds = model.predict(new_data)
        assert len(preds) == 2

        conf_int = model.predict_interval(new_data, interval_type='confidence')
        pred_int = model.predict_interval(new_data, interval_type='prediction')

        # Prediction interval must be strictly wider than confidence interval of the mean
        conf_width = conf_int['95% CI upper'] - conf_int['95% CI lower']
        pred_width = pred_int['95% CI upper'] - pred_int['95% CI lower']
        assert np.all(pred_width > conf_width)

    def test_frisch_waugh_lovell_decomposition(self, synthetic_ols_data):
        df = synthetic_ols_data
        # Full multivariate model
        full_model = LinearRegression.fit(X=df[['x1', 'x2']], y=df['y'], cov_type='HC3')

        # FWL partialling-out on x1
        tilde_D, tilde_y, fwl_bivariate = full_model.fwl_decompose('x1')

        # Slope of bivariate regression must exactly equal the multivariate regression coefficient
        beta_multi = float(full_model.params['x1'])
        beta_fwl = float(fwl_bivariate.params['x1_residualized'])
        assert np.isclose(beta_multi, beta_fwl, atol=1e-10)

    def test_ratio_residualize_and_regress(self):
        np.random.seed(42)
        n = 150
        confounder = np.random.normal(loc=50.0, scale=10.0, size=n)
        treatment = 0.5 * confounder + np.random.normal(0, 2, n)
        outcome = 100.0 + 3.0 * treatment + 2.0 * confounder + np.random.normal(0, 5, n)

        y_ratio = Ratio(pd.Series(outcome, name='revenue'))
        X_df = pd.DataFrame({'treatment': treatment, 'confounder': confounder})

        # 1. Ratio.regress
        reg_model = y_ratio.regress(X_df, cov_type='HC3')
        assert isinstance(reg_model, LinearRegression)
        assert np.isclose(reg_model.params['treatment'], 3.0, atol=0.5)

        # 2. Ratio.residualize: partial out confounder
        y_adj = y_ratio.residualize(pd.Series(confounder, name='confounder'), keep_mean=True)
        assert isinstance(y_adj, Ratio)
        assert len(y_adj) == n
        # Mean preserved
        assert np.isclose(y_adj.mean(), y_ratio.mean(), atol=1e-8)

        # Variance of adjusted outcome is strictly smaller than unadjusted outcome (due to partialling out confounder)
        assert y_adj.var() < y_ratio.var()

    def test_plots(self, synthetic_ols_data):
        df = synthetic_ols_data
        model = LinearRegression.fit(X=df[['x1', 'x2']], y=df['y'])

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        ax1 = model.plot_coefficients(ax=axes[0])
        ax2 = model.plot_fit(x_var='x1', ax=axes[1])
        ax3 = model.plot_fwl(treatment_var='x1', ax=axes[2])

        assert ax1 is axes[0]
        assert ax2 is axes[1]
        assert ax3 is axes[2]
        plt.close(fig)
