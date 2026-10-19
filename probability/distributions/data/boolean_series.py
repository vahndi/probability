from typing import Optional, TYPE_CHECKING

from matplotlib.axes import Axes
import numpy as np
from numpy import linspace, inf
from pandas import Series, DataFrame
from scipy.optimize import minimize
from scipy.special import betaln

from probability.distributions.data.boolean import Boolean
from probability.distributions.conjugate.beta_binomial_conjugate import BetaBinomialConjugate
from probability.distributions.continuous.beta import Beta
from probability.plotting import Color, add_v_density, new_axes
from probability.distributions.continuous.beta_series import BetaSeries
from probability.distributions.mixins.data.data_series_aggregate_mixins import \
    DataSeriesMinMixin, DataSeriesMaxMixin, DataSeriesMeanMixin, \
    DataSeriesModeMixin, DataSeriesMedianMixin, DataSeriesStdMixin, \
    DataSeriesVarMixin, DataSeriesSemMixin
from probability.distributions.mixins.data.data_series_mixin import \
    DataSeriesMixin
from probability.distributions.protocols import CategoricalDistribution


if TYPE_CHECKING:
    from probability.distributions.data.boolean_frame import BooleanFrame


class BooleanSeries(
    DataSeriesMixin,
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
    Series or dict of Boolean distributions.
    """

    def split_by(
            self,
            categorical: CategoricalDistribution
    ) -> 'BooleanFrame':
        """
        Split into a BooleanSeries on different values of the given categorical
        distribution.

        :param categorical: Distribution to split on
        """
        bools = []
        for self_cat in self.keys():
            self_cat_dist: Boolean = self._data[self_cat]
            for other_cat in categorical.categories:
                self_other_dist = self_cat_dist.filter_to(
                    categorical.keep(other_cat)
                )
                bools.append({
                    self.name: self_cat,
                    categorical.name: other_cat,
                    'value': self_other_dist
                })
        bools_frame = DataFrame(bools).pivot(
            index=categorical.name,
            columns=self.name,
            values='value'
        ).reindex(index=categorical.categories, columns=self.keys())
        from probability.distributions.data.boolean_frame import BooleanFrame
        return BooleanFrame(bools_frame)

    def to_beta_series(self) -> BetaSeries:
        """
        Convert to a Beta distribution.
        """
        return BetaSeries(data=self._data.map(lambda b: b.to_beta()))

    def empirical_bayes_prior(self, method: str = 'mle') -> Beta:
        """
        Estimate the shared Beta(alpha, beta) hyperparameter prior across categories
        using Empirical Bayes.

        :param method: 'mle' (default, marginal Beta-Binomial MLE) or 'moments' (method of moments).
        :return: Fitted Beta prior distribution.
        """
        cats = self.keys()
        if len(cats) < 2:
            raise ValueError("Empirical Bayes prior requires at least 2 categories.")

        ks = []
        ns = []
        for cat in cats:
            b_dist: Boolean = self._data[cat]
            k = int((b_dist.data == True).sum())
            n = len(b_dist)
            if n > 0:
                ks.append(k)
                ns.append(n)

        if len(ks) < 2:
            raise ValueError("At least 2 non-empty categories are required.")

        k_arr = np.asarray(ks, dtype=float)
        n_arr = np.asarray(ns, dtype=float)
        p_arr = k_arr / n_arr

        # Grand mean
        p_bar = float(np.sum(k_arr) / np.sum(n_arr))
        p_bar = float(np.clip(p_bar, 1e-4, 1.0 - 1e-4))

        # Method of moments estimate
        var_p = float(np.var(p_arr, ddof=1)) if len(p_arr) > 1 else 0.0
        n_mean = float(np.mean(n_arr))
        binom_var = p_bar * (1.0 - p_bar) / n_mean

        if var_p > binom_var:
            s_est = (p_bar * (1.0 - p_bar) - var_p) / (var_p - binom_var)
            s_est = max(s_est, 0.5)
            alpha_0 = max(p_bar * s_est, 0.1)
            beta_0 = max((1.0 - p_bar) * s_est, 0.1)
        else:
            alpha_0 = max(p_bar * 10.0, 0.5)
            beta_0 = max((1.0 - p_bar) * 10.0, 0.5)

        if method == 'moments':
            return Beta(alpha=alpha_0, beta=beta_0)
        elif method == 'mle':
            # Negative log-marginal Beta-Binomial likelihood over log(alpha), log(beta)
            def neg_log_lik(theta):
                log_a, log_b = theta
                a = np.exp(log_a)
                b = np.exp(log_b)
                # sum_j [ betaln(k_j + a, n_j - k_j + b) - betaln(a, b) ]
                terms = betaln(k_arr + a, n_arr - k_arr + b) - betaln(a, b)
                val = np.sum(terms)
                if np.isnan(val) or np.isneginf(val):
                    return 1e12
                return -float(val)

            x0 = np.array([np.log(alpha_0), np.log(beta_0)])
            res = minimize(neg_log_lik, x0=x0, method='L-BFGS-B')
            if res.success:
                a_mle = float(np.exp(res.x[0]))
                b_mle = float(np.exp(res.x[1]))
                return Beta(alpha=a_mle, beta=b_mle)
            else:
                return Beta(alpha=alpha_0, beta=beta_0)
        else:
            raise ValueError(f"Unknown method '{method}'. Must be 'mle' or 'moments'.")

    def shrunk_posteriors(
            self,
            prior: Optional[Beta] = None,
            method: str = 'mle'
    ) -> BetaSeries:
        """
        Compute Empirical Bayes shrunk posterior Beta distributions for each category.

        Pulls small-sample noisy categories toward the fitted grand mean prior:
            p_j | D ~ Beta(alpha + k_j, beta + n_j - k_j)

        :param prior: Optional pre-specified Beta prior. If None, fitted via Empirical Bayes.
        :param method: 'mle' (default) or 'moments' for fitting prior.
        :return: BetaSeries containing posterior Beta distributions for all categories.
        """
        if prior is None:
            prior = self.empirical_bayes_prior(method=method)

        alpha_prior = prior.alpha
        beta_prior = prior.beta

        shrunk_betas = {}
        for cat in self.keys():
            b_dist: Boolean = self._data[cat]
            k = int((b_dist.data == True).sum())
            n = len(b_dist)
            shrunk_betas[cat] = Beta(
                alpha=alpha_prior + k,
                beta=beta_prior + (n - k)
            )

        series_data = Series(shrunk_betas, name=f"shrunk_{self.name}" if self.name else "shrunk_posterior")
        series_data.index.name = self.index.name
        return BetaSeries(series_data)

    def shrinkage_summary(
            self,
            hdi: float = 0.95,
            prior: Optional[Beta] = None,
            method: str = 'mle'
    ) -> DataFrame:
        """
        Tidy DataFrame comparing raw category sample proportions with Empirical Bayes
        shrunk posterior estimates and HDIs.
        """
        if prior is None:
            prior = self.empirical_bayes_prior(method=method)

        shrunk_series = self.shrunk_posteriors(prior=prior)
        cats = self.keys()

        rows = []
        for cat in cats:
            b_dist: Boolean = self._data[cat]
            k = int((b_dist.data == True).sum())
            n = len(b_dist)
            raw_mean = k / n if n > 0 else np.nan
            shrunk_dist: Beta = shrunk_series[cat]
            low, high = shrunk_dist.hdi(hdi)
            weight = n / (n + prior.alpha + prior.beta) if n > 0 else 0.0

            rows.append({
                'category': cat,
                'k': k,
                'n': n,
                'raw_mean': raw_mean,
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
            prior: Optional[Beta] = None,
            method: str = 'mle',
            order_by: str = 'raw_mean',
            title: Optional[str] = None,
            ax: Optional[Axes] = None,
            **kwargs
    ) -> Axes:
        """
        Visualize Empirical Bayes shrinkage across categories.

        Plots raw category conversion rates alongside shrunk posterior means and HDIs,
        with arrows indicating the directional shrinkage toward the prior grand mean.

        :param hdi: Highest Density Interval width (default 0.95).
        :param prior: Optional pre-fitted Beta prior.
        :param method: 'mle' or 'moments' for Empirical Bayes estimation.
        :param order_by: 'raw_mean' (default), 'n', or 'original' to order categories.
        :param title: Optional custom title.
        :param ax: Optional matplotlib Axes.
        """
        ax = ax or new_axes()
        if prior is None:
            prior = self.empirical_bayes_prior(method=method)

        summary_df = self.shrinkage_summary(hdi=hdi, prior=prior)
        if order_by == 'raw_mean':
            summary_df = summary_df.sort_values('raw_mean')
        elif order_by == 'n':
            summary_df = summary_df.sort_values('n')

        cats = list(summary_df.index)
        n_cats = len(cats)
        y_positions = np.arange(n_cats)

        grand_mean = prior.mean()

        # Plot grand mean reference line
        ax.axvline(
            grand_mean, color='gray', linestyle='--', alpha=0.7,
            label=f'Prior Grand Mean ({grand_mean:.3f})'
        )

        raw_means = summary_df['raw_mean'].values
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
            raw_means, y_positions,
            color='tab:red', marker='x', s=50, zorder=3,
            label='Raw Sample Rate'
        )

        # Connect raw to shrunk with arrows
        for y, raw, shrunk in zip(y_positions, raw_means, shrunk_means):
            ax.annotate(
                '', xy=(shrunk, y), xytext=(raw, y),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2, alpha=0.7)
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels([f"{c} (n={summary_df.loc[c, 'n']})" for c in cats])
        ax.set_xlabel('Probability / Conversion Rate')
        ax.set_title(
            title or f"Empirical Bayes Shrinkage (Prior: Beta(α={prior.alpha:.2f}, β={prior.beta:.2f}))"
        )
        ax.legend(loc='best')
        return ax

    def plot_prob_densities(
            self,
            hdi: float = 0.95,
            width: float = 0.8,
            num_segments: int = 100,
            color: Color = 'k',
            color_min: Optional[Color] = None,
            color_mean: Optional[Color] = None,
            edge_color: Optional[Color] = None,
            ax: Optional[Axes] = None
    ) -> Axes:
        """
        Plot conditional probability densities of the data, split by the
        categories of an Ordinal or Nominal distribution.

        :param hdi: Highest Density Interval width for each distribution.
        :param width: Width of each density bar.
        :param num_segments: Number of segments to plot per density.
        :param color: Color for the densest part of each distribution.
        :param color_min: Color for the sparsest part of each distribution,
                          if different to color.
        :param color_mean: Color for mean data markers.
        :param edge_color: Optional color for the edge of each density bar.
        :param ax: Optional matplotlib Axes to plot on.
        """
        ax = ax or new_axes()

        cats = list(self.keys())
        n_cats = len(self._data)
        yy_min, yy_max = inf, -inf
        for c, category in enumerate(self.keys()):
            # fit distribution and find limits for HDI
            cat_data: Series = self._data[category].data
            cat_dist = BetaBinomialConjugate.infer_posterior(cat_data)
            # cat_dist = Beta.fit(data=cat_ratio_data)
            y_min, y_max = cat_dist.hdi(hdi)
            yy_min, yy_max = min(y_min, yy_min), max(y_max, yy_max)
            # plot density
            add_v_density(
                ax=ax,
                x=c + 1,
                y_to_z=cat_dist.pdf().at(
                    linspace(y_min, y_max, num_segments + 1)),
                color=color, color_min=color_min, edge_color=edge_color,
                width=width
            )
            # plot descriptive statistics lines
            if color_mean is not None:
                mean = cat_data.mean()
                ax.plot([c + 0.55, c + 1.45], [mean, mean], color=color_mean)
        # labels
        ax.set_title(
            f'{hdi: .0%} HDIs of $p(' +
            r'p_{' + self.name + r'}' +
            f'|{self.index.name})$'
        )
        ax.set_xlabel(self.index.name)
        ax.set_ylabel(r'$p_{' + self.name + r'}$')
        # axes
        ax.set_xlim(0, n_cats + 1)
        yy_range = yy_max - yy_min
        ax.set_ylim(yy_min - yy_range * 0.05, yy_max + yy_range * 0.05)
        ax.set_yticks(linspace(0, 1, 11))
        ax.set_xticks(range(1, n_cats + 1))
        ax.set_xticklabels(cats)

        return ax

