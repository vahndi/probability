from math import sqrt
from typing import Optional, Union, Dict, List, Tuple, Any

from matplotlib.axes import Axes
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame, Series
from scipy import linalg
from scipy.stats import t as student_t, f as fisher_f

from probability.distributions.continuous.students_t import StudentsT
from probability.plotting import new_axes
from probability.utils import num_format


class LinearRegression(object):
    """
    Ordinary Least Squares (OLS) and Causal Adjustment Regression with
    heteroskedasticity-robust (HC0, HC1, HC2, HC3) and cluster-robust standard errors.

    Model:
        y = X β + ε

    Properties & Distributions:
        Coefficients β_j are exposed as first-class Student's t sampling distributions:
            β_j ~ StudentsT(nu = df_resid, mu = β̂_j, sigma = SE(β̂_j))
        enabling exact interval estimation, hypothesis testing (β_j > 0), and plotting.

    Supports the Frisch-Waugh-Lovell (FWL) partialling-out theorem via `.fwl_decompose()`.
    """

    def __init__(
            self,
            params: Series,
            cov_params: DataFrame,
            bse: Series,
            residuals: Series,
            fitted_values: Series,
            cov_type: str,
            df_resid: float,
            df_model: int,
            n_obs: int,
            rsquared: float,
            rsquared_adj: float,
            f_statistic: float,
            f_pvalue: float,
            scale: float,
            feature_names: List[str],
            fit_intercept: bool,
            X_orig: DataFrame,
            y_orig: Series,
            cluster_name: Optional[str] = None
    ):
        self._params: Series = params
        self._cov_params: DataFrame = cov_params
        self._bse: Series = bse
        self._residuals: Series = residuals
        self._fitted_values: Series = fitted_values
        self._cov_type: str = cov_type
        self._df_resid: float = df_resid
        self._df_model: int = df_model
        self._n_obs: int = n_obs
        self._rsquared: float = rsquared
        self._rsquared_adj: float = rsquared_adj
        self._f_statistic: float = f_statistic
        self._f_pvalue: float = f_pvalue
        self._scale: float = scale
        self._feature_names: List[str] = feature_names
        self._fit_intercept: bool = fit_intercept
        self._X_orig: DataFrame = X_orig
        self._y_orig: Series = y_orig
        self._cluster_name: Optional[str] = cluster_name

    # region Properties

    @property
    def params(self) -> Series:
        """Estimated coefficient vector β̂."""
        return self._params

    @property
    def bse(self) -> Series:
        """Standard errors of coefficients SE(β̂)."""
        return self._bse

    @property
    def cov_params(self) -> DataFrame:
        """Variance-covariance matrix of estimated coefficients."""
        return self._cov_params

    @property
    def cov_type(self) -> str:
        """Type of covariance estimator used ('nonrobust', 'HC0', 'HC1', 'HC2', 'HC3', 'cluster')."""
        return self._cov_type

    @property
    def residuals(self) -> Series:
        """Model residuals: e = y - X β̂."""
        return self._residuals

    @property
    def fitted_values(self) -> Series:
        """Fitted values: ŷ = X β̂."""
        return self._fitted_values

    @property
    def df_resid(self) -> float:
        """Residual degrees of freedom (n - k for HC / nonrobust, G - 1 for cluster)."""
        return self._df_resid

    @property
    def df_model(self) -> int:
        """Model degrees of freedom (number of predictors excluding intercept)."""
        return self._df_model

    @property
    def n_obs(self) -> int:
        """Number of observations used in model fitting."""
        return self._n_obs

    @property
    def rsquared(self) -> float:
        """Coefficient of determination R²."""
        return self._rsquared

    @property
    def rsquared_adj(self) -> float:
        """Adjusted coefficient of determination."""
        return self._rsquared_adj

    @property
    def f_statistic(self) -> float:
        """Overall F-statistic."""
        return self._f_statistic

    @property
    def f_pvalue(self) -> float:
        """p-value for the overall F-statistic."""
        return self._f_pvalue

    @property
    def scale(self) -> float:
        """Residual variance s² = SSE / (n - k)."""
        return self._scale

    @property
    def feature_names(self) -> List[str]:
        """Names of features / coefficients."""
        return self._feature_names

    @property
    def tvalues(self) -> Series:
        """t-statistics for coefficients: t = β̂ / SE(β̂)."""
        return self._params / self._bse

    @property
    def pvalues(self) -> Series:
        """Two-tailed p-values for coefficients."""
        t_vals = np.abs(self.tvalues.values)
        p_vals = 2.0 * student_t.sf(t_vals, df=self._df_resid)
        return Series(p_vals, index=self._feature_names, name='p_value')

    # endregion

    # region Distributions

    def coefficient(self, name: str) -> StudentsT:
        """
        Return the exact Student's t sampling distribution for a specific coefficient.

        :param name: Feature / coefficient name.
        :return: StudentsT(nu = df_resid, mu = β̂_j, sigma = SE(β̂_j)).
        """
        if name not in self._feature_names:
            raise KeyError(f"Feature '{name}' not found. Available features: {self._feature_names}")
        dist = StudentsT(
            nu=self._df_resid,
            mu=float(self._params[name]),
            sigma=float(self._bse[name])
        )
        dist.with_label(name)
        return dist

    @property
    def coefficients(self) -> Dict[str, StudentsT]:
        """
        Dictionary of Student's t sampling distributions for all coefficients.
        """
        return {name: self.coefficient(name) for name in self._feature_names}

    # endregion

    # region Fitting

    @classmethod
    def fit(
            cls,
            X: Union[DataFrame, Series, ndarray, List[Any], Any],
            y: Union[Series, ndarray, List[Any], Any],
            cov_type: Optional[str] = None,
            cluster: Optional[Union[Series, ndarray, List[Any], Any]] = None,
            fit_intercept: bool = True
    ) -> 'LinearRegression':
        """
        Fit an Ordinary Least Squares (OLS) regression with optional robust standard errors.

        :param X: Design matrix / predictors (DataFrame, Series, ndarray, or distribution objects).
        :param y: Outcome variable (Series, Ratio, or ndarray).
        :param cov_type: Covariance estimator:
                         - 'HC3' (default when cluster is None): MacKinnon & White (1985) leverage-adjusted
                         - 'HC2': MacKinnon & White (1985)
                         - 'HC1': Stata default robust (n / (n - k) * HC0)
                         - 'HC0': White (1980) heteroskedasticity-consistent
                         - 'nonrobust': standard homoskedastic OLS
                         - 'cluster': cluster-robust sandwich covariance (auto-selected if cluster is provided)
        :param cluster: Cluster identifiers for cluster-robust standard errors.
        :param fit_intercept: Whether to include an intercept term (default True).
        :return: Fitted LinearRegression instance.
        """
        # 1. Format and align y
        if hasattr(y, 'data') and isinstance(y.data, Series):
            y_series = y.data.astype(float)
        elif isinstance(y, Series):
            y_series = y.astype(float)
        else:
            y_series = Series(np.asarray(y, dtype=float).ravel(), name='y')

        # 2. Format X
        if hasattr(X, 'data') and isinstance(X.data, Series):
            X_df = DataFrame({X.data.name or 'x1': X.data.astype(float)})
        elif isinstance(X, Series):
            X_df = DataFrame({X.name or 'x1': X.astype(float)})
        elif isinstance(X, DataFrame):
            X_df = X.copy().astype(float)
        elif isinstance(X, (list, tuple)) and len(X) > 0 and hasattr(X[0], 'data'):
            X_dict = {}
            for i, item in enumerate(X):
                name = getattr(item, 'name', None) or f'x{i+1}'
                X_dict[name] = item.data.astype(float)
            X_df = DataFrame(X_dict)
        else:
            X_arr = np.asarray(X, dtype=float)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            X_df = DataFrame(
                X_arr,
                columns=[f'x{i+1}' for i in range(X_arr.shape[1])]
            )

        # 3. Format cluster if provided
        cluster_series = None
        cluster_name = None
        if cluster is not None:
            if isinstance(cluster, Series):
                cluster_series = cluster
                cluster_name = cluster.name or 'cluster'
            else:
                cluster_series = Series(np.asarray(cluster), name='cluster')
                cluster_name = 'cluster'

        # 4. Align indices and drop missing values
        combined_dict = {'__y__': y_series}
        for col in X_df.columns:
            combined_dict[col] = X_df[col]
        if cluster_series is not None:
            combined_dict['__cluster__'] = cluster_series

        combined_df = DataFrame(combined_dict).dropna()
        n = len(combined_df)
        if n == 0:
            raise ValueError("No valid observations remaining after dropping missing values.")

        y_clean = combined_df['__y__']
        X_clean = combined_df[X_df.columns]
        cluster_clean = combined_df['__cluster__'] if cluster_series is not None else None

        X_orig = X_clean.copy()
        y_orig = y_clean.copy()

        # 5. Add intercept if requested
        if fit_intercept:
            if 'intercept' not in X_clean.columns:
                X_clean.insert(0, 'intercept', 1.0)

        feature_names = list(X_clean.columns)
        k = len(feature_names)
        if n <= k:
            raise ValueError(f"Number of observations (n={n}) must exceed number of features (k={k}).")

        X_mat = X_clean.values.astype(float)
        y_vec = y_clean.values.astype(float)

        # 6. OLS Point Estimates: beta = (X^T X)^(-1) X^T y
        XtX = X_mat.T @ X_mat
        try:
            XtX_inv = linalg.inv(XtX)
        except linalg.LinAlgError:
            XtX_inv = linalg.pinv(XtX)

        Xty = X_mat.T @ y_vec
        beta = XtX_inv @ Xty

        # Fitted values and residuals
        y_hat = X_mat @ beta
        e = y_vec - y_hat
        sse = float(np.sum(e ** 2))
        s2 = sse / float(n - k)

        # R-squared and F-statistic
        if fit_intercept and 'intercept' in feature_names:
            y_bar = float(np.mean(y_vec))
            sst = float(np.sum((y_vec - y_bar) ** 2))
            df_model = k - 1
        else:
            sst = float(np.sum(y_vec ** 2))
            df_model = k

        if sst > 1e-12:
            r2 = max(0.0, min(1.0, 1.0 - sse / sst))
            r2_adj = 1.0 - (1.0 - r2) * ((n - 1) / (n - k))
        else:
            r2 = 0.0
            r2_adj = 0.0

        if df_model > 0 and (1.0 - r2) > 1e-12:
            f_stat = float((r2 / df_model) / ((1.0 - r2) / (n - k)))
            f_pval = float(fisher_f.sf(f_stat, dfn=df_model, dfd=n - k))
        else:
            f_stat = np.nan
            f_pval = np.nan

        # 7. Covariance Matrix Estimation
        if cov_type is None:
            cov_type = 'cluster' if cluster_clean is not None else 'HC3'

        cov_type_lower = cov_type.lower()
        df_resid = float(n - k)

        if cov_type_lower in ('nonrobust', 'ols', 'homoskedastic'):
            cov_type_clean = 'nonrobust'
            V = s2 * XtX_inv

        elif cov_type_lower in ('hc0', 'hc1', 'hc2', 'hc3'):
            cov_type_clean = cov_type.upper()
            # Efficient hat matrix diagonal: h_i = sum((X * XtX_inv) * X, axis=1)
            h = np.sum((X_mat @ XtX_inv) * X_mat, axis=1)
            h = np.minimum(h, 1.0 - 1e-10)

            if cov_type_clean == 'HC0':
                w = e ** 2
            elif cov_type_clean == 'HC1':
                w = (float(n) / float(n - k)) * (e ** 2)
            elif cov_type_clean == 'HC2':
                w = (e ** 2) / (1.0 - h)
            elif cov_type_clean == 'HC3':
                w = (e ** 2) / ((1.0 - h) ** 2)
            else:
                w = e ** 2

            # Meat: X^T diag(w) X = (X * sqrt(w))^T (X * sqrt(w))
            sqrt_w = np.sqrt(np.maximum(w, 0.0))[:, np.newaxis]
            X_w = X_mat * sqrt_w
            meat = X_w.T @ X_w
            V = XtX_inv @ meat @ XtX_inv

        elif cov_type_lower in ('cluster', 'clustered'):
            cov_type_clean = 'cluster'
            if cluster_clean is None:
                raise ValueError("cov_type='cluster' requires 'cluster' identifiers to be provided.")

            cluster_labels, cluster_indices = np.unique(cluster_clean.values, return_inverse=True)
            G = len(cluster_labels)
            if G <= 1:
                raise ValueError(f"Cluster covariance requires at least 2 distinct clusters, found {G}.")

            # Compute meat: sum_g (X_g^T e_g)(X_g^T e_g)^T
            meat = np.zeros((k, k), dtype=float)
            for g in range(G):
                mask = (cluster_indices == g)
                u_g = X_mat[mask].T @ e[mask]
                meat += np.outer(u_g, u_g)

            # Standard small-sample cluster correction: c = ((n - 1) / (n - k)) * (G / (G - 1))
            c = ((n - 1) / (n - k)) * (G / (G - 1))
            V = c * (XtX_inv @ meat @ XtX_inv)
            df_resid = float(G - 1)

        else:
            raise ValueError(
                f"Unknown cov_type '{cov_type}'. Must be one of: "
                "'nonrobust', 'HC0', 'HC1', 'HC2', 'HC3', 'cluster'."
            )

        se = np.sqrt(np.maximum(np.diag(V), 1e-12))

        params_s = Series(beta, index=feature_names, name='coefficient')
        cov_df = DataFrame(V, index=feature_names, columns=feature_names)
        bse_s = Series(se, index=feature_names, name='std_err')
        res_s = Series(e, index=y_clean.index, name='residuals')
        fit_s = Series(y_hat, index=y_clean.index, name='fitted_values')

        return cls(
            params=params_s,
            cov_params=cov_df,
            bse=bse_s,
            residuals=res_s,
            fitted_values=fit_s,
            cov_type=cov_type_clean,
            df_resid=df_resid,
            df_model=df_model,
            n_obs=n,
            rsquared=r2,
            rsquared_adj=r2_adj,
            f_statistic=f_stat,
            f_pvalue=f_pval,
            scale=s2,
            feature_names=feature_names,
            fit_intercept=fit_intercept,
            X_orig=X_orig,
            y_orig=y_orig,
            cluster_name=cluster_name
        )

    # endregion

    # region Summary & Inspection

    def summary(self, alpha: float = 0.05) -> DataFrame:
        """
        Produce a tidy regression summary table with estimates, standard errors,
        t-statistics, p-values, and confidence / HDI intervals.

        :param alpha: Significance level for confidence interval (default 0.05 for 95% CI).
        :return: DataFrame indexed by feature name.
        """
        rows = []
        crit_t = float(student_t.ppf(1.0 - alpha / 2.0, df=self._df_resid))
        pct = int((1.0 - alpha) * 100)

        for name in self._feature_names:
            coef = float(self._params[name])
            se = float(self._bse[name])
            t_val = coef / se if se > 0 else np.nan
            p_val = 2.0 * float(student_t.sf(abs(t_val), df=self._df_resid)) if not np.isnan(t_val) else np.nan
            low = coef - crit_t * se
            high = coef + crit_t * se

            rows.append({
                'feature': name,
                'coef': coef,
                'std_err': se,
                't_stat': t_val,
                'p_value': p_val,
                f'{pct}% CI lower': low,
                f'{pct}% CI upper': high,
                'df': self._df_resid
            })

        return DataFrame(rows).set_index('feature')

    # endregion

    # region Prediction

    def predict(self, X_new: Union[DataFrame, Series, ndarray, Any]) -> Series:
        """
        Generate point predictions ŷ_new = X_new β̂.

        :param X_new: DataFrame or ndarray of new predictor values.
        :return: Series of point predictions.
        """
        X_mat, index = self._format_X_new(X_new)
        y_pred = X_mat @ self._params.values
        return Series(y_pred, index=index, name='prediction')

    def predict_interval(
            self,
            X_new: Union[DataFrame, Series, ndarray, Any],
            alpha: float = 0.05,
            interval_type: str = 'confidence'
    ) -> DataFrame:
        """
        Compute mean predictions and confidence or prediction intervals.

        :param X_new: New predictor values.
        :param alpha: Significance level (default 0.05 for 95% interval).
        :param interval_type: 'confidence' (interval for mean response E[Y|X]) or
                              'prediction' (interval for a new individual observation Y_*).
        :return: DataFrame with columns ['mean', 'se', 'ci_lower', 'ci_upper'].
        """
        X_mat, index = self._format_X_new(X_new)
        y_hat = X_mat @ self._params.values
        V = self._cov_params.values

        # Variance of mean prediction: x_* V x_*^T
        pred_var = np.sum((X_mat @ V) * X_mat, axis=1)

        if interval_type == 'prediction':
            # Add residual variance s^2
            pred_var += self._scale
        elif interval_type != 'confidence':
            raise ValueError(f"Unknown interval_type '{interval_type}'. Must be 'confidence' or 'prediction'.")

        pred_se = np.sqrt(np.maximum(pred_var, 1e-12))
        crit_t = float(student_t.ppf(1.0 - alpha / 2.0, df=self._df_resid))
        pct = int((1.0 - alpha) * 100)

        return DataFrame({
            'mean': y_hat,
            'se': pred_se,
            f'{pct}% CI lower': y_hat - crit_t * pred_se,
            f'{pct}% CI upper': y_hat + crit_t * pred_se
        }, index=index)

    def _format_X_new(self, X_new: Union[DataFrame, Series, ndarray, Any]) -> Tuple[ndarray, Any]:
        """Format and validate X_new for predictions."""
        if hasattr(X_new, 'data') and isinstance(X_new.data, Series):
            X_df = DataFrame({X_new.data.name or 'x1': X_new.data.astype(float)})
        elif isinstance(X_new, Series):
            X_df = DataFrame({X_new.name or 'x1': X_new.astype(float)})
        elif isinstance(X_new, DataFrame):
            X_df = X_new.copy().astype(float)
        else:
            X_arr = np.asarray(X_new, dtype=float)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            # Match non-intercept predictors
            preds = [col for col in self._feature_names if col != 'intercept']
            if X_arr.shape[1] == len(preds):
                X_df = DataFrame(X_arr, columns=preds)
            elif X_arr.shape[1] == len(self._feature_names):
                X_df = DataFrame(X_arr, columns=self._feature_names)
            else:
                raise ValueError(
                    f"X_new shape {X_arr.shape} does not match expected predictors {self._feature_names}."
                )

        if self._fit_intercept and 'intercept' not in X_df.columns:
            X_df.insert(0, 'intercept', 1.0)

        # Ensure all required features are present
        missing = [col for col in self._feature_names if col not in X_df.columns]
        if missing:
            raise ValueError(f"Missing predictor columns in X_new: {missing}")

        X_df = X_df[self._feature_names]
        return X_df.values.astype(float), X_df.index

    # endregion

    # region Frisch-Waugh-Lovell Decomposition

    def fwl_decompose(
            self,
            treatment_var: str
    ) -> Tuple[Series, Series, 'LinearRegression']:
        """
        Execute Frisch-Waugh-Lovell (FWL) partialling-out decomposition for a specific treatment variable.

        FWL Theorem states that in the multivariate regression:
            y = β_T * D + X_other * γ + ε
        the estimated coefficient β̂_T is mathematically identical to the coefficient from
        the bivariate regression of the residualized outcome on the residualized treatment:
            ỹ = β_T * D̃ + u
        where:
            D̃ = D - X_other (X_other^T X_other)^(-1) X_other^T D
            ỹ = y - X_other (X_other^T X_other)^(-1) X_other^T y

        :param treatment_var: The name of the treatment variable to isolate.
        :return: (tilde_D, tilde_y, bivariate_model)
                 where tilde_D and tilde_y are Series of residuals, and bivariate_model
                 is a fitted LinearRegression instance.
        """
        if treatment_var not in self._feature_names:
            raise KeyError(f"Treatment variable '{treatment_var}' not found in model predictors: {self._feature_names}")
        if treatment_var == 'intercept':
            raise ValueError("Cannot perform FWL decomposition on the intercept.")

        # Covariates to partial out: all features except treatment_var
        other_covariates = [col for col in self._X_orig.columns if col != treatment_var]

        # 1. Regress Treatment D on other covariates
        D_series = self._X_orig[treatment_var]
        if other_covariates:
            model_D = LinearRegression.fit(
                X=self._X_orig[other_covariates],
                y=D_series,
                fit_intercept=self._fit_intercept,
                cov_type='nonrobust'
            )
            tilde_D = model_D.residuals
        else:
            # If no other covariates, demean if intercept
            tilde_D = D_series - D_series.mean() if self._fit_intercept else D_series

        tilde_D.name = f"{treatment_var}_residualized"

        # 2. Regress Outcome y on other covariates
        if other_covariates:
            model_y = LinearRegression.fit(
                X=self._X_orig[other_covariates],
                y=self._y_orig,
                fit_intercept=self._fit_intercept,
                cov_type='nonrobust'
            )
            tilde_y = model_y.residuals
        else:
            tilde_y = self._y_orig - self._y_orig.mean() if self._fit_intercept else self._y_orig

        tilde_y.name = f"{self._y_orig.name or 'y'}_residualized"

        # 3. Fit bivariate regression: tilde_y on tilde_D without intercept
        bivariate_model = LinearRegression.fit(
            X=tilde_D,
            y=tilde_y,
            fit_intercept=False,
            cov_type=self._cov_type
        )

        return tilde_D, tilde_y, bivariate_model

    # endregion

    # region Plotting

    def plot_coefficients(
            self,
            hdi: float = 0.95,
            ax: Optional[Axes] = None,
            title: Optional[str] = None
    ) -> Axes:
        """
        Forest plot of regression coefficients with confidence / HDI intervals and reference line at 0.

        :param hdi: Confidence level (default 0.95).
        :param ax: Matplotlib Axes.
        :param title: Optional custom title.
        :return: Axes.
        """
        ax = ax or new_axes()
        summary_df = self.summary(alpha=1.0 - hdi)
        pct = int(hdi * 100)

        names = list(summary_df.index)
        means = summary_df['coef'].values
        lowers = summary_df[f'{pct}% CI lower'].values
        uppers = summary_df[f'{pct}% CI upper'].values

        y_positions = np.arange(len(names))
        xerr = np.vstack([means - lowers, uppers - means])

        ax.errorbar(
            means, y_positions, xerr=xerr,
            fmt='o', color='darkblue', ecolor='teal',
            elinewidth=2, capsize=4, capthick=1.5
        )

        ax.axvline(0, color='gray', linestyle='--', alpha=0.7)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(names)
        ax.set_xlabel('Coefficient Value')
        ax.set_title(title or f'OLS Coefficients ({pct}% CI, {self._cov_type} SE)')
        ax.grid(True, linestyle=':', alpha=0.5)

        return ax

    def plot_fit(
            self,
            x_var: Optional[str] = None,
            hdi: float = 0.95,
            n_grid: int = 100,
            ax: Optional[Axes] = None
    ) -> Axes:
        """
        Plot observed scatter points, fitted regression line, and shaded confidence band.

        :param x_var: Predictor variable name to plot against. Defaults to first predictor.
        :param hdi: Confidence level (default 0.95).
        :param n_grid: Number of grid evaluation points.
        :param ax: Matplotlib Axes.
        :return: Axes.
        """
        ax = ax or new_axes()
        predictors = [f for f in self._X_orig.columns if f != 'intercept']
        if x_var is None:
            if len(predictors) != 1:
                raise ValueError(f"Please specify x_var from available predictors: {predictors}")
            x_var = predictors[0]

        x_vals = self._X_orig[x_var].values
        y_vals = self._y_orig.values

        ax.scatter(x_vals, y_vals, color='darkgray', alpha=0.7, label='Observed Data')

        x_min, x_max = x_vals.min(), x_vals.max()
        x_span = x_max - x_min
        grid_x = np.linspace(x_min - 0.05 * x_span, x_max + 0.05 * x_span, n_grid)

        grid_dict = {x_var: grid_x}
        for col in predictors:
            if col != x_var:
                grid_dict[col] = float(self._X_orig[col].mean())
        grid_df = DataFrame(grid_dict)

        intervals = self.predict_interval(grid_df, alpha=1.0 - hdi, interval_type='confidence')
        pct = int(hdi * 100)
        low_col = f'{pct}% CI lower'
        high_col = f'{pct}% CI upper'

        ax.plot(grid_x, intervals['mean'], color='navy', linewidth=2, label='Fitted Line')
        ax.fill_between(
            grid_x, intervals[low_col], intervals[high_col],
            color='skyblue', alpha=0.4, label=f'{pct}% Confidence Band'
        )

        ax.set_xlabel(x_var)
        ax.set_ylabel(self._y_orig.name or 'y')
        ax.set_title(f'OLS Fit: {self._y_orig.name or "y"} ~ {x_var}')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.5)

        return ax

    def plot_fwl(
            self,
            treatment_var: str,
            hdi: float = 0.95,
            ax: Optional[Axes] = None,
            title: Optional[str] = None
    ) -> Axes:
        """
        Visualize the Frisch-Waugh-Lovell partialled-out regression for a treatment variable.

        Plots the scatter of residualized outcome ỹ vs residualized treatment D̃,
        along with the causal slope β̂_T and confidence band.

        :param treatment_var: The name of the treatment variable.
        :param hdi: Confidence level (default 0.95).
        :param ax: Matplotlib Axes.
        :param title: Optional title.
        :return: Axes.
        """
        ax = ax or new_axes()
        tilde_D, tilde_y, b_model = self.fwl_decompose(treatment_var)
        pct = int(hdi * 100)

        # Scatter of residualized values
        ax.scatter(tilde_D.values, tilde_y.values, color='slategray', alpha=0.6, label='Partialled Observations')

        # Regression line
        d_min, d_max = tilde_D.min(), tilde_D.max()
        span = d_max - d_min
        grid_d = np.linspace(d_min - 0.05 * span, d_max + 0.05 * span, 100)

        grid_df = DataFrame({tilde_D.name: grid_d})
        intervals = b_model.predict_interval(grid_df, alpha=1.0 - hdi, interval_type='confidence')

        beta_val = float(b_model.params[tilde_D.name])
        se_val = float(b_model.bse[tilde_D.name])

        ax.plot(
            grid_d, intervals['mean'], color='crimson', linewidth=2,
            label=f'FWL Causal Slope β̂ = {beta_val:.3f} (SE={se_val:.3f})'
        )
        ax.fill_between(
            grid_d, intervals[f'{pct}% CI lower'], intervals[f'{pct}% CI upper'],
            color='salmon', alpha=0.3, label=f'{pct}% Confidence Band'
        )

        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel(f'Residualized Treatment: {tilde_D.name}')
        ax.set_ylabel(f'Residualized Outcome: {tilde_y.name}')
        ax.set_title(title or f'FWL Partialling-Out: {self._y_orig.name or "y"} on {treatment_var}')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.5)

        return ax

    # endregion

    def __str__(self) -> str:
        coef_str = ', '.join([
            f'{name}={num_format(self._params[name], 3)}'
            for name in self._feature_names
        ])
        return f'LinearRegression(n={self._n_obs}, R²={self._rsquared:.3f}, cov={self._cov_type}, β=[{coef_str}])'

    def __repr__(self) -> str:
        return self.__str__()


# Alias OLS for convenience and familiarity
OLS = LinearRegression
