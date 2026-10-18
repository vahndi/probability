from math import sqrt
from typing import Optional, Union, Dict, List, Tuple, Any

from matplotlib.axes import Axes
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame, Series
from scipy.optimize import minimize
from scipy.stats import t as student_t

from probability.distributions.continuous.students_t import StudentsT
from probability.models.linear_regression import LinearRegression
from probability.plotting import new_axes


def _fit_logistic(
        X_mat: ndarray,
        t_vec: ndarray,
        l2_reg: float = 1e-5
) -> Tuple[ndarray, ndarray]:
    """
    Fit binary logistic regression via penalized maximum likelihood.
    Returns (coefficients, predicted_probabilities).
    """
    n, k = X_mat.shape

    def neg_log_lik(gamma):
        # Clip linear predictor to prevent overflow in exp
        z = np.clip(X_mat @ gamma, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        p_safe = np.clip(p, 1e-12, 1.0 - 1e-12)
        ll = np.sum(t_vec * np.log(p_safe) + (1.0 - t_vec) * np.log(1.0 - p_safe))
        # L2 penalty excluding intercept (assumed at index 0)
        penalty = 0.5 * l2_reg * np.sum(gamma[1:] ** 2) if k > 1 else 0.0
        return float(-ll + penalty)

    def grad(gamma):
        z = np.clip(X_mat @ gamma, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        err = p - t_vec
        g = X_mat.T @ err
        if k > 1:
            pen_g = np.zeros_like(gamma)
            pen_g[1:] = l2_reg * gamma[1:]
            g += pen_g
        return g

    # Initial guess: log-odds of treatment rate for intercept, 0 for others
    gamma0 = np.zeros(k, dtype=float)
    t_rate = np.clip(float(np.mean(t_vec)), 0.01, 0.99)
    gamma0[0] = np.log(t_rate / (1.0 - t_rate))

    res = minimize(neg_log_lik, x0=gamma0, jac=grad, method='L-BFGS-B')
    opt_gamma = res.x if res.success else gamma0
    z = np.clip(X_mat @ opt_gamma, -30.0, 30.0)
    p_hat = 1.0 / (1.0 + np.exp(-z))
    return opt_gamma, p_hat


def _prepare_causal_data(
        treatment: Union[Series, ndarray, List[Any], Any],
        outcome: Union[Series, ndarray, List[Any], Any],
        covariates: Union[DataFrame, Series, ndarray, List[Any], Any]
) -> Tuple[Series, Series, DataFrame]:
    """Align indices, clean missing values, and validate causal inputs."""
    # 1. Format Treatment
    if hasattr(treatment, 'data') and isinstance(treatment.data, Series):
        t_s = treatment.data.astype(float)
    elif isinstance(treatment, Series):
        t_s = treatment.astype(float)
    else:
        t_s = Series(np.asarray(treatment, dtype=float).ravel(), name='treatment')

    # 2. Format Outcome
    if hasattr(outcome, 'data') and isinstance(outcome.data, Series):
        y_s = outcome.data.astype(float)
    elif isinstance(outcome, Series):
        y_s = outcome.astype(float)
    else:
        y_s = Series(np.asarray(outcome, dtype=float).ravel(), name='outcome')

    # 3. Format Covariates
    if hasattr(covariates, 'data') and isinstance(covariates.data, Series):
        X_df = DataFrame({covariates.data.name or 'x1': covariates.data.astype(float)})
    elif isinstance(covariates, Series):
        X_df = DataFrame({covariates.name or 'x1': covariates.astype(float)})
    elif isinstance(covariates, DataFrame):
        X_df = covariates.copy().astype(float)
    else:
        X_arr = np.asarray(covariates, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        X_df = DataFrame(X_arr, columns=[f'x{i+1}' for i in range(X_arr.shape[1])])

    # Align indices
    combined_dict = {'__t__': t_s, '__y__': y_s}
    for col in X_df.columns:
        combined_dict[col] = X_df[col]
    combined_df = DataFrame(combined_dict).dropna()

    t_clean = combined_df['__t__']
    y_clean = combined_df['__y__']
    X_clean = combined_df[X_df.columns]

    # Check binary treatment
    u_vals = set(np.unique(t_clean.values))
    if not u_vals.issubset({0.0, 1.0}):
        raise ValueError(f"Treatment must be binary (0 and 1), found values: {u_vals}")
    if len(u_vals) < 2:
        raise ValueError("Both treated (1) and control (0) units must be present in the data.")

    return t_clean, y_clean, X_clean


class IPW(object):
    """
    Inverse Probability Weighting (IPW) causal effect estimator with propensity score
    diagnostics and Love plot covariate balance checks.

    Supports ATE (Average Treatment Effect) and ATT (Average Treatment Effect on the Treated)
    using normalized (Hajek) weights.
    """

    def __init__(
            self,
            estimate: float,
            std_err: float,
            estimand: str,
            propensity_scores: Series,
            weights: Series,
            treatment: Series,
            outcome: Series,
            covariates: DataFrame,
            n_obs: int
    ):
        self._estimate: float = estimate
        self._std_err: float = std_err
        self._estimand: str = estimand
        self._propensity_scores: Series = propensity_scores
        self._weights: Series = weights
        self._treatment: Series = treatment
        self._outcome: Series = outcome
        self._covariates: DataFrame = covariates
        self._n_obs: int = n_obs

    @property
    def estimate(self) -> float:
        """Causal treatment effect point estimate."""
        return self._estimate

    @property
    def std_err(self) -> float:
        """Standard error of the causal estimate."""
        return self._std_err

    @property
    def estimand(self) -> str:
        """Target estimand: 'ATE' or 'ATT'."""
        return self._estimand

    @property
    def propensity_scores(self) -> Series:
        """Estimated propensity scores P(T = 1 | X)."""
        return self._propensity_scores

    @property
    def weights(self) -> Series:
        """Calculated IPW weights."""
        return self._weights

    @property
    def n_obs(self) -> int:
        """Number of observations."""
        return self._n_obs

    @property
    def effect_distribution(self) -> StudentsT:
        """
        Exact Student's t sampling distribution for the causal treatment effect.
        """
        dist = StudentsT(
            nu=float(self._n_obs - 1),
            mu=self._estimate,
            sigma=self._std_err
        )
        dist.with_label(f'{self._estimand}_IPW')
        return dist

    @classmethod
    def fit(
            cls,
            treatment: Union[Series, ndarray, List[Any], Any],
            outcome: Union[Series, ndarray, List[Any], Any],
            covariates: Union[DataFrame, Series, ndarray, List[Any], Any],
            estimand: str = 'ATE',
            clip_bounds: Tuple[float, float] = (0.01, 0.99),
            n_bootstraps: int = 0,
            seed: Optional[int] = None
    ) -> 'IPW':
        """
        Fit an IPW model to estimate causal effects.

        :param treatment: Binary treatment (0 or 1).
        :param outcome: Continuous outcome metric.
        :param covariates: Confounder variables.
        :param estimand: 'ATE' (default) or 'ATT'.
        :param clip_bounds: (min_p, max_p) propensity score clipping bounds.
        :param n_bootstraps: Number of bootstrap iterations for SE calculation (0 for analytical).
        :param seed: Random seed for bootstrapping.
        :return: Fitted IPW instance.
        """
        t_s, y_s, X_df = _prepare_causal_data(treatment, outcome, covariates)
        estimand = estimand.upper()
        if estimand not in ('ATE', 'ATT'):
            raise ValueError(f"Unknown estimand '{estimand}'. Must be 'ATE' or 'ATT'.")

        n = len(t_s)
        X_with_int = X_df.copy()
        X_with_int.insert(0, 'intercept', 1.0)
        X_mat = X_with_int.values.astype(float)
        t_vec = t_s.values.astype(float)
        y_vec = y_s.values.astype(float)

        # 1. Fit Propensity Score
        _, p_hat = _fit_logistic(X_mat, t_vec)
        p_hat = np.clip(p_hat, clip_bounds[0], clip_bounds[1])

        # 2. Compute IPW Weights
        if estimand == 'ATE':
            w = (t_vec / p_hat) + ((1.0 - t_vec) / (1.0 - p_hat))
            # Normalized Hajek estimator
            y1_hat = np.sum((t_vec * y_vec) / p_hat) / np.sum(t_vec / p_hat)
            y0_hat = np.sum(((1.0 - t_vec) * y_vec) / (1.0 - p_hat)) / np.sum((1.0 - t_vec) / (1.0 - p_hat))
            tau_hat = float(y1_hat - y0_hat)
            # Influence function approximation for analytical SE
            psi = (t_vec * (y_vec - y1_hat) / p_hat) - ((1.0 - t_vec) * (y_vec - y0_hat) / (1.0 - p_hat))
            se = float(np.std(psi, ddof=1) / sqrt(n))
        else:  # ATT
            w = t_vec + (1.0 - t_vec) * (p_hat / (1.0 - p_hat))
            w_ctl = (1.0 - t_vec) * (p_hat / (1.0 - p_hat))
            y1_hat = np.sum(t_vec * y_vec) / np.sum(t_vec)
            y0_hat = np.sum(w_ctl * y_vec) / np.sum(w_ctl)
            tau_hat = float(y1_hat - y0_hat)
            p_T = np.mean(t_vec)
            psi = (t_vec * (y_vec - y1_hat) / p_T) - (w_ctl * (y_vec - y0_hat) / p_T)
            se = float(np.std(psi, ddof=1) / sqrt(n))

        # Optional bootstrap SE
        if n_bootstraps > 0:
            rng = np.random.RandomState(seed)
            boot_taus = []
            for _ in range(n_bootstraps):
                idx = rng.choice(n, size=n, replace=True)
                X_b = X_mat[idx]
                t_b = t_vec[idx]
                y_b = y_vec[idx]
                if len(np.unique(t_b)) < 2:
                    continue
                try:
                    _, p_b = _fit_logistic(X_b, t_b)
                    p_b = np.clip(p_b, clip_bounds[0], clip_bounds[1])
                    if estimand == 'ATE':
                        y1_b = np.sum((t_b * y_b) / p_b) / np.sum(t_b / p_b)
                        y0_b = np.sum(((1.0 - t_b) * y_b) / (1.0 - p_b)) / np.sum((1.0 - t_b) / (1.0 - p_b))
                        boot_taus.append(y1_b - y0_b)
                    else:
                        w_ctl_b = (1.0 - t_b) * (p_b / (1.0 - p_b))
                        y1_b = np.sum(t_b * y_b) / np.sum(t_b)
                        y0_b = np.sum(w_ctl_b * y_b) / np.sum(w_ctl_b)
                        boot_taus.append(y1_b - y0_b)
                except Exception:
                    continue
            if boot_taus:
                se = float(np.std(boot_taus, ddof=1))

        p_s = Series(p_hat, index=t_s.index, name='propensity_score')
        w_s = Series(w, index=t_s.index, name='weight')

        return cls(
            estimate=tau_hat,
            std_err=se,
            estimand=estimand,
            propensity_scores=p_s,
            weights=w_s,
            treatment=t_s,
            outcome=y_s,
            covariates=X_df,
            n_obs=n
        )

    def summary(self, alpha: float = 0.05) -> DataFrame:
        """Tidy summary table of the estimated causal treatment effect."""
        crit_t = float(student_t.ppf(1.0 - alpha / 2.0, df=self._n_obs - 1))
        t_stat = self._estimate / self._std_err if self._std_err > 0 else np.nan
        p_val = 2.0 * float(student_t.sf(abs(t_stat), df=self._n_obs - 1)) if not np.isnan(t_stat) else np.nan
        pct = int((1.0 - alpha) * 100)

        df = DataFrame([{
            'estimand': self._estimand,
            'estimate': self._estimate,
            'std_err': self._std_err,
            't_stat': t_stat,
            'p_value': p_val,
            f'{pct}% CI lower': self._estimate - crit_t * self._std_err,
            f'{pct}% CI upper': self._estimate + crit_t * self._std_err,
            'n_obs': self._n_obs
        }]).set_index('estimand')
        return df

    def balance_summary(self) -> DataFrame:
        """
        Compute unweighted and weighted standardized mean differences (Cohen's d / normalized diff)
        for each covariate before and after IPW weighting.
        """
        rows = []
        t_mask = (self._treatment.values == 1.0)
        c_mask = ~t_mask
        w = self._weights.values

        for col in self._covariates.columns:
            x = self._covariates[col].values.astype(float)
            x_t = x[t_mask]
            x_c = x[c_mask]

            # Unweighted standardized mean difference
            m_t, m_c = np.mean(x_t), np.mean(x_c)
            v_t, v_c = np.var(x_t, ddof=1), np.var(x_c, ddof=1)
            s_unw = sqrt(max((v_t + v_c) / 2.0, 1e-12))
            diff_unw = float((m_t - m_c) / s_unw)

            # Weighted standardized mean difference
            w_t = w[t_mask]
            w_c = w[c_mask]
            wm_t = np.sum(w_t * x_t) / np.sum(w_t)
            wm_c = np.sum(w_c * x_c) / np.sum(w_c)
            wv_t = np.sum(w_t * (x_t - wm_t) ** 2) / np.sum(w_t)
            wv_c = np.sum(w_c * x_c - wm_c) ** 2 / np.sum(w_c)
            s_w = sqrt(max((wv_t + wv_c) / 2.0, 1e-12))
            diff_w = float((wm_t - wm_c) / s_w)

            rows.append({
                'covariate': col,
                'unweighted_diff': diff_unw,
                'weighted_diff': diff_w,
                'abs_unweighted': abs(diff_unw),
                'abs_weighted': abs(diff_w),
                'balanced': abs(diff_w) < 0.1
            })

        return DataFrame(rows).set_index('covariate')

    def love_plot(
            self,
            threshold: float = 0.1,
            ax: Optional[Axes] = None,
            title: Optional[str] = None
    ) -> Axes:
        """
        Generate a Love plot comparing absolute standardized mean differences
        before (unweighted) and after (weighted) IPW adjustments.

        :param threshold: Vertical threshold line for balance (default 0.1).
        :param ax: Matplotlib Axes.
        :param title: Optional custom title.
        :return: Axes.
        """
        ax = ax or new_axes()
        bal_df = self.balance_summary().sort_values('abs_unweighted')
        covars = list(bal_df.index)
        y_pos = np.arange(len(covars))

        ax.scatter(bal_df['abs_unweighted'], y_pos, color='firebrick', marker='o', s=60, label='Unweighted (Raw)')
        ax.scatter(bal_df['abs_weighted'], y_pos, color='seagreen', marker='s', s=60, label=f'Weighted ({self._estimand})')

        # Connect pairs with a light gray line
        for i in range(len(covars)):
            ax.plot(
                [bal_df['abs_unweighted'].iloc[i], bal_df['abs_weighted'].iloc[i]],
                [y_pos[i], y_pos[i]],
                color='gray', linestyle=':', alpha=0.6
            )

        ax.axvline(threshold, color='black', linestyle='--', alpha=0.7, label=f'Threshold ({threshold})')
        ax.axvline(0.0, color='gray', linestyle='-', alpha=0.3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(covars)
        ax.set_xlabel('Absolute Standardized Mean Difference')
        ax.set_title(title or f'Love Plot: Covariate Balance ({self._estimand} IPW)')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.4)

        return ax

    def plot_propensity(
            self,
            ax: Optional[Axes] = None,
            title: Optional[str] = None
    ) -> Axes:
        """
        Plot propensity score distributions for treated vs. control groups to check common support / overlap.
        """
        ax = ax or new_axes()
        t_mask = (self._treatment.values == 1.0)
        p_treated = self._propensity_scores.values[t_mask]
        p_control = self._propensity_scores.values[~t_mask]

        bins = np.linspace(0.0, 1.0, 25)
        ax.hist(p_control, bins=bins, alpha=0.5, density=True, color='skyblue', label='Control (T=0)')
        ax.hist(p_treated, bins=bins, alpha=0.5, density=True, color='salmon', label='Treated (T=1)')

        ax.set_xlabel('Propensity Score P(T=1 | X)')
        ax.set_ylabel('Density')
        ax.set_title(title or 'Propensity Score Overlap (Common Support)')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.4)

        return ax

    def __str__(self) -> str:
        return f'IPW(estimand={self._estimand}, estimate={self._estimate:.3f}, SE={self._std_err:.3f}, n={self._n_obs})'

    def __repr__(self) -> str:
        return self.__str__()


class AIPW(object):
    """
    Augmented Inverse Probability Weighting (AIPW) / Doubly Robust causal estimator.

    Combines an outcome regression model E[Y | T, X] with a propensity score model P(T=1 | X).
    Possesses the Doubly Robust property: consistent and unbiased if EITHER the outcome model
    OR the propensity model is correctly specified.
    """

    def __init__(
            self,
            estimate: float,
            std_err: float,
            estimand: str,
            scores: Series,
            propensity_scores: Series,
            treatment: Series,
            outcome: Series,
            covariates: DataFrame,
            n_obs: int,
            outcome_model_t1: LinearRegression,
            outcome_model_t0: LinearRegression
    ):
        self._estimate: float = estimate
        self._std_err: float = std_err
        self._estimand: str = estimand
        self._scores: Series = scores
        self._propensity_scores: Series = propensity_scores
        self._treatment: Series = treatment
        self._outcome: Series = outcome
        self._covariates: DataFrame = covariates
        self._n_obs: int = n_obs
        self._model_t1: LinearRegression = outcome_model_t1
        self._model_t0: LinearRegression = outcome_model_t0

    @property
    def estimate(self) -> float:
        """Doubly robust causal treatment effect estimate."""
        return self._estimate

    @property
    def std_err(self) -> float:
        """Standard error of the doubly robust estimate."""
        return self._std_err

    @property
    def estimand(self) -> str:
        """Target estimand: 'ATE' or 'ATT'."""
        return self._estimand

    @property
    def scores(self) -> Series:
        """Individual doubly robust influence scores."""
        return self._scores

    @property
    def propensity_scores(self) -> Series:
        """Estimated propensity scores."""
        return self._propensity_scores

    @property
    def n_obs(self) -> int:
        """Number of observations."""
        return self._n_obs

    @property
    def effect_distribution(self) -> StudentsT:
        """
        Exact Student's t sampling distribution for the causal effect.
        """
        dist = StudentsT(
            nu=float(self._n_obs - 1),
            mu=self._estimate,
            sigma=self._std_err
        )
        dist.with_label(f'{self._estimand}_AIPW')
        return dist

    @classmethod
    def fit(
            cls,
            treatment: Union[Series, ndarray, List[Any], Any],
            outcome: Union[Series, ndarray, List[Any], Any],
            covariates: Union[DataFrame, Series, ndarray, List[Any], Any],
            estimand: str = 'ATE',
            clip_bounds: Tuple[float, float] = (0.01, 0.99),
            n_bootstraps: int = 0,
            seed: Optional[int] = None
    ) -> 'AIPW':
        """
        Fit an Augmented IPW (Doubly Robust) causal model.

        :param treatment: Binary treatment (0 or 1).
        :param outcome: Continuous outcome metric.
        :param covariates: Confounder variables.
        :param estimand: 'ATE' (default) or 'ATT'.
        :param clip_bounds: Propensity score clipping bounds.
        :param n_bootstraps: Optional number of bootstrap iterations for SE calculation.
        :param seed: Random seed for bootstrapping.
        :return: Fitted AIPW instance.
        """
        t_s, y_s, X_df = _prepare_causal_data(treatment, outcome, covariates)
        estimand = estimand.upper()
        if estimand not in ('ATE', 'ATT'):
            raise ValueError(f"Unknown estimand '{estimand}'. Must be 'ATE' or 'ATT'.")

        n = len(t_s)
        X_with_int = X_df.copy()
        X_with_int.insert(0, 'intercept', 1.0)
        X_mat = X_with_int.values.astype(float)
        t_vec = t_s.values.astype(float)
        y_vec = y_s.values.astype(float)

        # 1. Fit Propensity Model
        _, p_hat = _fit_logistic(X_mat, t_vec)
        p_hat = np.clip(p_hat, clip_bounds[0], clip_bounds[1])

        # 2. Fit Outcome Models: mu_1(X) on treated units, mu_0(X) on control units
        t_mask = (t_vec == 1.0)
        c_mask = ~t_mask

        model_t1 = LinearRegression.fit(
            X=X_df[t_mask],
            y=y_s[t_mask],
            cov_type='nonrobust'
        )
        model_t0 = LinearRegression.fit(
            X=X_df[c_mask],
            y=y_s[c_mask],
            cov_type='nonrobust'
        )

        mu1_hat = model_t1.predict(X_df).values
        mu0_hat = model_t0.predict(X_df).values

        # 3. Doubly Robust Scores
        if estimand == 'ATE':
            # psi_1 = mu_1(X) + T * (Y - mu_1(X)) / p_hat
            psi1 = mu1_hat + (t_vec * (y_vec - mu1_hat)) / p_hat
            # psi_0 = mu_0(X) + (1 - T) * (Y - mu_0(X)) / (1 - p_hat)
            psi0 = mu0_hat + ((1.0 - t_vec) * (y_vec - mu0_hat)) / (1.0 - p_hat)
            scores = psi1 - psi0
            tau_hat = float(np.mean(scores))
            se = float(np.std(scores, ddof=1) / sqrt(n))
        else:  # ATT
            p_T = float(np.mean(t_vec))
            scores = (t_vec * (y_vec - mu0_hat) / p_T) - (((1.0 - t_vec) * p_hat * (y_vec - mu0_hat)) / (p_T * (1.0 - p_hat)))
            tau_hat = float(np.mean(scores))
            se = float(np.std(scores, ddof=1) / sqrt(n))

        if n_bootstraps > 0:
            rng = np.random.RandomState(seed)
            boot_taus = []
            for _ in range(n_bootstraps):
                idx = rng.choice(n, size=n, replace=True)
                X_b = X_df.iloc[idx]
                t_b = t_vec[idx]
                y_b = y_vec[idx]
                if len(np.unique(t_b)) < 2:
                    continue
                try:
                    X_int_b = X_b.copy()
                    X_int_b.insert(0, 'intercept', 1.0)
                    _, p_b = _fit_logistic(X_int_b.values.astype(float), t_b)
                    p_b = np.clip(p_b, clip_bounds[0], clip_bounds[1])

                    m1_b = LinearRegression.fit(X=X_b[t_b == 1.0], y=y_b[t_b == 1.0], cov_type='nonrobust')
                    m0_b = LinearRegression.fit(X=X_b[t_b == 0.0], y=y_b[t_b == 0.0], cov_type='nonrobust')
                    mu1_b = m1_b.predict(X_b).values
                    mu0_b = m0_b.predict(X_b).values

                    if estimand == 'ATE':
                        b_psi1 = mu1_b + (t_b * (y_b - mu1_b)) / p_b
                        b_psi0 = mu0_b + ((1.0 - t_b) * (y_b - mu0_b)) / (1.0 - p_b)
                        boot_taus.append(float(np.mean(b_psi1 - b_psi0)))
                    else:
                        p_Tb = float(np.mean(t_b))
                        b_scores = (t_b * (y_b - mu0_b) / p_Tb) - (((1.0 - t_b) * p_b * (y_b - mu0_b)) / (p_Tb * (1.0 - p_b)))
                        boot_taus.append(float(np.mean(b_scores)))
                except Exception:
                    continue
            if boot_taus:
                se = float(np.std(boot_taus, ddof=1))

        scores_s = Series(scores, index=t_s.index, name='doubly_robust_score')
        p_s = Series(p_hat, index=t_s.index, name='propensity_score')

        return cls(
            estimate=tau_hat,
            std_err=se,
            estimand=estimand,
            scores=scores_s,
            propensity_scores=p_s,
            treatment=t_s,
            outcome=y_s,
            covariates=X_df,
            n_obs=n,
            outcome_model_t1=model_t1,
            outcome_model_t0=model_t0
        )

    def summary(self, alpha: float = 0.05) -> DataFrame:
        """Tidy summary table of the estimated doubly robust treatment effect."""
        crit_t = float(student_t.ppf(1.0 - alpha / 2.0, df=self._n_obs - 1))
        t_stat = self._estimate / self._std_err if self._std_err > 0 else np.nan
        p_val = 2.0 * float(student_t.sf(abs(t_stat), df=self._n_obs - 1)) if not np.isnan(t_stat) else np.nan
        pct = int((1.0 - alpha) * 100)

        df = DataFrame([{
            'estimand': self._estimand,
            'estimate': self._estimate,
            'std_err': self._std_err,
            't_stat': t_stat,
            'p_value': p_val,
            f'{pct}% CI lower': self._estimate - crit_t * self._std_err,
            f'{pct}% CI upper': self._estimate + crit_t * self._std_err,
            'n_obs': self._n_obs
        }]).set_index('estimand')
        return df

    def __str__(self) -> str:
        return f'AIPW(estimand={self._estimand}, estimate={self._estimate:.3f}, SE={self._std_err:.3f}, n={self._n_obs})'

    def __repr__(self) -> str:
        return self.__str__()


# Aliases
DoublyRobust = AIPW
DoublyRobustEstimator = AIPW
IPWEstimator = IPW
