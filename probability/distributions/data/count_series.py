from typing import Optional, Dict

from matplotlib.axes import Axes
import numpy as np
from numpy.random import choice
from pandas import Series, DataFrame, concat
from scipy.optimize import minimize
from scipy.special import gammaln

from probability.distributions.continuous.gamma import Gamma
from probability.distributions.continuous.gamma_series import GammaSeries
from probability.distributions.discrete.negative_binomial import NegativeBinomial
from probability.distributions.data.count import Count
from probability.plotting import (
    Color,
    add_rectangle,
    cross_fade,
    format_axis_integer,
    new_axes,
    set_alpha,
)
from probability.distributions.mixins.data.data_series_aggregate_mixins import \
    DataSeriesMinMixin, DataSeriesMaxMixin, DataSeriesMeanMixin, \
    DataSeriesModeMixin, DataSeriesMedianMixin, DataSeriesStdMixin, \
    DataSeriesVarMixin, DataSeriesSemMixin
from probability.distributions.mixins.data.data_series_category_mixins import \
    DataSeriesCountsMixin, DataSeriesPMFsMixin
from probability.distributions.mixins.data.data_series_mixin import \
    DataSeriesMixin, DataSeriesTrendMixin


class CountSeries(
    DataSeriesMixin,
    DataSeriesTrendMixin,
    DataSeriesCountsMixin,
    DataSeriesPMFsMixin,
    DataSeriesMinMixin,
    DataSeriesMaxMixin,
    DataSeriesMeanMixin,
    DataSeriesModeMixin,
    DataSeriesMedianMixin,
    DataSeriesStdMixin,
    DataSeriesVarMixin,
    DataSeriesSemMixin,
    object
):
    """
    Series or dict of Count distributions.
    """

    def unsplit(self) -> Count:
        """
        Return a Count with all the data for each Count in the
        CountSeries concatenated.
        """
        return Count(
            data=concat([
                self._data[key].data for key in self.keys()
            ])
        )

    def _sample_series(self, unsplit_dist: Count, n_x: Series) -> 'CountSeries':
        p_y = unsplit_dist.pmf()
        y = p_y.index.to_list()
        s_x = {
            k: Count(Series(data=choice(a=y, size=n_x[k], p=p_y)))
            for k in self.keys()
        }
        return CountSeries(s_x)

    def empirical_bayes_prior(self, method: str = 'mle') -> Gamma:
        """
        Estimate the shared Gamma(alpha, beta) hyperparameter prior across count categories
        using Empirical Bayes.

        :param method: 'mle' (default, marginal Negative-Binomial MLE) or 'moments' (method of moments).
        :return: Fitted Gamma prior distribution.
        """
        cats = self.keys()
        if len(cats) < 2:
            raise ValueError("Empirical Bayes prior requires at least 2 categories.")

        ks = []
        ns = []
        for cat in cats:
            c_dist: Count = self._data[cat]
            k_sum = int(c_dist.data.sum())
            n = len(c_dist)
            if n > 0:
                ks.append(k_sum)
                ns.append(n)

        if len(ks) < 2:
            raise ValueError("At least 2 non-empty categories are required.")

        k_arr = np.asarray(ks, dtype=float)
        n_arr = np.asarray(ns, dtype=float)
        y_rates = k_arr / n_arr

        # Weighted grand mean rate lambda
        lambda_bar = float(np.sum(k_arr) / np.sum(n_arr))
        lambda_bar = max(lambda_bar, 1e-4)

        # Method of moments estimate
        var_y = float(np.var(y_rates, ddof=1)) if len(y_rates) > 1 else 0.0
        n_mean = float(np.mean(n_arr))
        expected_poisson_var = lambda_bar / n_mean

        if var_y > expected_poisson_var:
            gamma_var = var_y - expected_poisson_var
            # Gamma(alpha, beta): mean = alpha/beta, var = alpha/beta^2 = mean/beta
            beta_0 = max(lambda_bar / gamma_var, 0.01)
            alpha_0 = max(lambda_bar * beta_0, 0.01)
        else:
            beta_0 = 1.0
            alpha_0 = max(lambda_bar, 0.1)

        if method == 'moments':
            return Gamma(alpha=alpha_0, beta=beta_0)
        elif method == 'mle':
            # Negative log-marginal likelihood for Negative Binomial over log(alpha), log(beta)
            def neg_log_lik(theta):
                log_a, log_b = theta
                a = np.exp(log_a)
                b = np.exp(log_b)
                # sum_j [ gammaln(K_j + a) - gammaln(a) + a * ln(b) - (K_j + a) * ln(b + n_j) ]
                terms = (
                    gammaln(k_arr + a) - gammaln(a) +
                    a * np.log(b) - (k_arr + a) * np.log(b + n_arr)
                )
                val = np.sum(terms)
                if np.isnan(val) or np.isneginf(val):
                    return 1e12
                return -float(val)

            x0 = np.array([np.log(alpha_0), np.log(beta_0)])
            res = minimize(neg_log_lik, x0=x0, method='L-BFGS-B')
            if res.success:
                a_mle = float(np.exp(res.x[0]))
                b_mle = float(np.exp(res.x[1]))
                return Gamma(alpha=a_mle, beta=b_mle)
            else:
                return Gamma(alpha=alpha_0, beta=beta_0)
        else:
            raise ValueError(f"Unknown method '{method}'. Must be 'mle' or 'moments'.")

    def shrunk_posteriors(
            self,
            prior: Optional[Gamma] = None,
            method: str = 'mle'
    ) -> GammaSeries:
        """
        Compute Empirical Bayes shrunk posterior Gamma distributions for each category.

        Pulls small-sample noisy event counts toward the fitted grand mean prior:
            lambda_j | D ~ Gamma(alpha + K_j, beta + n_j)

        :param prior: Optional pre-specified Gamma prior. If None, fitted via Empirical Bayes.
        :param method: 'mle' (default) or 'moments' for fitting prior.
        :return: GammaSeries containing posterior Gamma distributions for all categories.
        """
        if prior is None:
            prior = self.empirical_bayes_prior(method=method)

        alpha_prior = prior.alpha
        beta_prior = prior.beta

        shrunk_gammas = {}
        for cat in self.keys():
            c_dist: Count = self._data[cat]
            k_sum = int(c_dist.data.sum())
            n = len(c_dist)
            shrunk_gammas[cat] = Gamma(
                alpha=alpha_prior + k_sum,
                beta=beta_prior + n
            )

        series_data = Series(shrunk_gammas, name=f"shrunk_{self.name}" if self.name else "shrunk_posterior")
        series_data.index.name = self.index.name
        return GammaSeries(series_data)

    def predictive_distributions(
            self,
            prior: Optional[Gamma] = None,
            time_exposure: float = 1.0,
            method: str = 'mle'
    ) -> Dict[str, NegativeBinomial]:
        """
        Compute Negative Binomial predictive distributions for future counts over `time_exposure`.

        :param prior: Optional pre-specified Gamma prior.
        :param time_exposure: Future observation interval length (default 1.0).
        :param method: 'mle' or 'moments'.
        :return: Dictionary mapping category name -> NegativeBinomial distribution.
        """
        if prior is None:
            prior = self.empirical_bayes_prior(method=method)

        preds = {}
        for cat in self.keys():
            c_dist: Count = self._data[cat]
            k_sum = int(c_dist.data.sum())
            n = len(c_dist)
            post_alpha = prior.alpha + k_sum
            post_beta = prior.beta + n
            # Scipy NegativeBinomial(r, p) where p is probability of failure (target in [0, 1])
            p_val = time_exposure / (post_beta + time_exposure)
            preds[cat] = NegativeBinomial(r=post_alpha, p=p_val)
        return preds

    def shrinkage_summary(
            self,
            hdi: float = 0.95,
            prior: Optional[Gamma] = None,
            method: str = 'mle'
    ) -> DataFrame:
        """
        Tidy DataFrame comparing raw category sample rates with Empirical Bayes
        shrunk posterior estimates and HDIs.
        """
        if prior is None:
            prior = self.empirical_bayes_prior(method=method)

        shrunk_series = self.shrunk_posteriors(prior=prior)
        cats = self.keys()

        rows = []
        for cat in cats:
            c_dist: Count = self._data[cat]
            k_sum = int(c_dist.data.sum())
            n = len(c_dist)
            raw_rate = k_sum / n if n > 0 else np.nan
            shrunk_dist: Gamma = shrunk_series[cat]
            low, high = shrunk_dist.hdi(hdi)
            weight = n / (n + prior.beta) if n > 0 else 0.0

            rows.append({
                'category': cat,
                'total_count': k_sum,
                'exposure': n,
                'raw_rate': raw_rate,
                'shrunk_mean': shrunk_dist.mean(),
                'shrunk_std': shrunk_dist.std(),
                f'{hdi:.0%} HDI lower': low,
                f'{hdi:.0%} HDI upper': high,
                'weight_data': weight
            })

        df = DataFrame(rows).set_index('category')
        return df

    def plot_shrinkage(
            self,
            hdi: float = 0.95,
            prior: Optional[Gamma] = None,
            method: str = 'mle',
            order_by: str = 'raw_rate',
            title: Optional[str] = None,
            ax: Optional[Axes] = None,
            **kwargs
    ) -> Axes:
        """
        Visualize Empirical Bayes shrinkage across count categories.

        Plots raw category event rates alongside shrunk posterior means and HDIs,
        with arrows indicating the directional shrinkage toward the prior grand mean.

        :param hdi: Highest Density Interval width (default 0.95).
        :param prior: Optional pre-fitted Gamma prior.
        :param method: 'mle' or 'moments' for Empirical Bayes estimation.
        :param order_by: 'raw_rate' (default), 'exposure', or 'original' to order categories.
        :param title: Optional custom title.
        :param ax: Optional matplotlib Axes.
        """
        ax = ax or new_axes()
        if prior is None:
            prior = self.empirical_bayes_prior(method=method)

        summary_df = self.shrinkage_summary(hdi=hdi, prior=prior)
        if order_by == 'raw_rate':
            summary_df = summary_df.sort_values('raw_rate')
        elif order_by == 'exposure':
            summary_df = summary_df.sort_values('exposure')

        cats = list(summary_df.index)
        n_cats = len(cats)
        y_positions = np.arange(n_cats)

        grand_mean = prior.mean()

        # Plot grand mean reference line
        ax.axvline(
            grand_mean, color='gray', linestyle='--', alpha=0.7,
            label=f'Prior Grand Mean ({grand_mean:.3f})'
        )

        raw_rates = summary_df['raw_rate'].values
        shrunk_means = summary_df['shrunk_mean'].values
        lowers = summary_df[f'{hdi:.0%} HDI lower'].values
        uppers = summary_df[f'{hdi:.0%} HDI upper'].values

        # Plot error bars for shrunk posterior HDIs
        xerr_low = shrunk_means - lowers
        xerr_high = uppers - shrunk_means
        ax.errorbar(
            shrunk_means, y_positions,
            xerr=[xerr_low, xerr_high],
            fmt='o', color='tab:blue', ecolor='tab:blue',
            elinewidth=1.8, capsize=3.5, markersize=6,
            label='Shrunk Posterior Mean & HDI'
        )

        # Plot raw points
        ax.scatter(
            raw_rates, y_positions,
            color='tab:red', marker='x', s=50, zorder=3,
            label='Raw Sample Rate'
        )

        # Connect raw to shrunk with arrows
        for y, raw, shrunk in zip(y_positions, raw_rates, shrunk_means):
            ax.annotate(
                '', xy=(shrunk, y), xytext=(raw, y),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2, alpha=0.7)
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels([f"{c} (n={summary_df.loc[c, 'exposure']})" for c in cats])
        ax.set_xlabel('Event Rate (λ)')
        ax.set_title(
            title or f"Empirical Bayes Shrinkage (Prior: Gamma(α={prior.alpha:.2f}, β={prior.beta:.2f}))"
        )
        ax.legend(loc='best')
        return ax

    def plot_bars(
            self,
            conditional: bool = False,
            max_pct: Optional[float] = None,
            width: float = 0.8,
            height: float = 0.8,
            color: Color = 'k',
            color_min: Optional[Color] = None,
            alpha_min: float = 0.0,
            edge_color: Optional[Color] = 'grey',
            ax: Optional[Axes] = None
    ) -> Axes:
        """
        Plot a set of bars for each Series, shaded by the count at each discrete
        value.

        :param conditional: Whether the shading for each Series should be
                            independent of all the others.
        :param max_pct: Highest percentile of each Series to show a bar for.
        :param width: Width of each set of bars.
        :param height: Height of each bar, centered about the count value.
        :param color: Color of each bar.
        :param color_min: Optional different color for the sparsest bars.
        :param alpha_min: Alpha value for the sparsest bars.
        :param edge_color: Optional edge color of each bar.
        :param ax: Optional matplotlib Axes instance.
        """
        ax = ax or new_axes()
        dist: Count
        # find highest count of all Series
        max_count = max(
            counts.max() for counts in self.counts()
        )
        # find highest value for y-axis scaling
        if max_pct is None:
            # find highest value of all Series
            max_val = self.max().max()
        else:
            max_val = 0
        # plot each Series
        for x, (ix, dist) in enumerate(self._data.items()):
            dist_items = dist.counts()
            if max_pct:
                max_dist_val = dist.data.quantile(max_pct)
                dist_items = dist_items.loc[dist_items.index <= max_dist_val]
                max_val = max(max_val, max_dist_val)
            if conditional:
                max_count = dist_items.max()
            for dist_value, dist_count in dist_items.items():
                coords = dict(
                    width=width, height=height,
                    x_center=(1 + x),
                    y_center=dist_value
                )
                if color_min is not None:
                    from_color = set_alpha(color_min, alpha_min)
                else:
                    from_color = set_alpha(color, alpha_min)
                add_rectangle(
                    ax=ax,
                    **coords,
                    color=cross_fade(
                        from_color=from_color,
                        to_color=color,
                        amount=dist_count / max_count
                    )
                )
                if edge_color is not None:
                    add_rectangle(
                        ax=ax,
                        **coords,
                        fill=False, color=None, edge_color=edge_color
                    )
        ax.set_xlim(0.5, len(self._data) + 0.5)
        ax.set_ylim(self.min().min() - 1, max_val + 1)
        ax.set_xticks(range(1, len(self._data) + 1))
        ax.set_xticklabels(self._data.index.to_list())

        format_axis_integer(ax.yaxis)

        return ax


if __name__ == '__main__':

    from probability.distributions import Poisson
    dists = {}
    for i in range(5, 10):
        dists[i] = Count(Poisson(i).rvs(10_000))
    cs = CountSeries(Series(dists))
    cs.plot_bars(color='red').show()
    cs.plot_bars(color='red', conditional=True).show()
    cs.plot_bars(color='red', color_min='yellow', conditional=True).show()
    cs.plot_bars(color='red', color_min='yellow',
                 alpha_min=0.5, conditional=True).show()
