from typing import Union, Optional, Iterable, Dict, List, Any

from matplotlib.axes import Axes
import numpy as np
from numpy import clip
from pandas import Series, DataFrame, concat
from scipy.optimize import minimize
from seaborn import kdeplot

from probability.distributions.continuous.normal import Normal
from probability.distributions.continuous.normal_series import NormalSeries
from probability.distributions.data.ratio import Ratio
from probability.plotting import (
    Color,
    add_rectangle,
    format_axis_integer,
    new_axes,
)
from probability.distributions.mixins.data.data_series_aggregate_mixins import \
    DataSeriesMinMixin, DataSeriesMaxMixin, DataSeriesMeanMixin, \
    DataSeriesStdMixin, DataSeriesVarMixin, DataSeriesSemMixin
from probability.distributions.mixins.data.data_series_mixin import \
    DataSeriesMixin, DataSeriesTrendMixin


class RatioSeries(
    DataSeriesMixin,
    DataSeriesTrendMixin,
    DataSeriesMinMixin,
    DataSeriesMaxMixin,
    DataSeriesMeanMixin,
    DataSeriesStdMixin,
    DataSeriesVarMixin,
    DataSeriesSemMixin,
    object
):
    """
    Series or dict of Ratio distributions.
    """

    @staticmethod
    def from_data_frame(
            data: DataFrame,
            ratio: str,
            split_by: str
    ) -> 'RatioSeries':
        """
        Create a RatioSeries containing a column of ratio data, and a column of
        discrete data to split by.

        :param data: DataFrame to use.
        :param ratio: Name of the column with ratio data.
        :param split_by: Name of the column with discrete data.
        """
        split_ratios = {}
        for unique in sorted(data[split_by].unique()):
            split_ratios[unique] = Ratio(data.loc[
                data[split_by] == unique,
                ratio
            ].rename(f'{ratio}|{split_by}={unique}'))
        return RatioSeries(split_ratios)

    def to_normal(self) -> NormalSeries:
        """
        Infer a Series of Normal distributions, one for each Ratio distribution.
        """
        return NormalSeries(
            Series({
                key: self._data[key].to_normal()
                for key in self.keys()
            }, name=self.name)
        )

    def empirical_bayes_prior(self, method: str = 'mle') -> Normal:
        """
        Estimate the shared Normal(mu_0, tau_0^2) prior across groups using Empirical Bayes
        (James-Stein / Random Effects).

        :param method: 'mle' (default, marginal Gaussian likelihood) or 'moments' (DerSimonian-Laird).
        :return: Fitted Normal prior distribution for group means.
        """
        cats = self.keys()
        if len(cats) < 2:
            raise ValueError("Empirical Bayes prior requires at least 2 categories.")

        means = []
        ns = []
        vars_ = []
        for cat in cats:
            r_dist: Ratio = self._data[cat]
            arr = np.asarray(r_dist.data, dtype=float)
            n = len(arr)
            if n > 0:
                m = float(np.mean(arr))
                v = float(np.var(arr, ddof=1)) if n > 1 else 0.0
                means.append(m)
                ns.append(n)
                vars_.append(v)

        if len(means) < 2:
            raise ValueError("At least 2 non-empty categories are required.")

        y_bar = np.asarray(means, dtype=float)
        n_arr = np.asarray(ns, dtype=float)
        s2_arr = np.asarray(vars_, dtype=float)

        # Calculate pooled residual variance for groups with small n
        df_total = np.sum(np.maximum(n_arr - 1.0, 0.0))
        if df_total > 0:
            pooled_s2 = float(np.sum(np.maximum(n_arr - 1.0, 0.0) * s2_arr) / df_total)
        else:
            pooled_s2 = float(np.var(y_bar, ddof=1)) if len(y_bar) > 1 else 1.0
        pooled_s2 = max(pooled_s2, 1e-6)

        # Standard error of each group mean: sigma_j^2 = s_j^2 / n_j
        se2_arr = np.zeros_like(y_bar)
        for i in range(len(y_bar)):
            if n_arr[i] > 1 and s2_arr[i] > 0:
                se2_arr[i] = s2_arr[i] / n_arr[i]
            else:
                se2_arr[i] = pooled_s2 / n_arr[i]
        se2_arr = np.maximum(se2_arr, 1e-8)

        # DerSimonian-Laird method of moments initial estimate
        M = len(y_bar)
        w = 1.0 / se2_arr
        mu_w = float(np.sum(w * y_bar) / np.sum(w))
        Q = float(np.sum(w * ((y_bar - mu_w) ** 2)))
        denom = float(np.sum(w) - (np.sum(w ** 2) / np.sum(w)))

        if Q > (M - 1) and denom > 0:
            tau0_sq_mom = float((Q - (M - 1)) / denom)
        else:
            tau0_sq_mom = float(np.var(y_bar, ddof=1) * 0.5) if M > 1 else 1.0
        tau0_sq_mom = max(tau0_sq_mom, 1e-6)

        # Precision-weighted grand mean under tau0_sq_mom
        weights_prior = 1.0 / (tau0_sq_mom + se2_arr)
        mu0_mom = float(np.sum(weights_prior * y_bar) / np.sum(weights_prior))

        if method == 'moments':
            return Normal(mu=mu0_mom, sigma=np.sqrt(tau0_sq_mom))
        elif method == 'mle':
            # Negative log-marginal Gaussian likelihood over (mu_0, log(tau_0^2))
            def neg_log_lik(theta):
                mu_val, log_tau2 = theta
                tau2 = np.exp(log_tau2)
                tot_var = tau2 + se2_arr
                ll = -0.5 * np.sum(np.log(2.0 * np.pi * tot_var) + ((y_bar - mu_val) ** 2) / tot_var)
                if np.isnan(ll) or np.isneginf(ll):
                    return 1e12
                return -float(ll)

            x0 = np.array([mu0_mom, np.log(tau0_sq_mom)])
            res = minimize(neg_log_lik, x0=x0, method='L-BFGS-B')
            if res.success:
                mu_mle = float(res.x[0])
                tau2_mle = float(np.exp(res.x[1]))
                tau2_mle = max(tau2_mle, 1e-8)
                return Normal(mu=mu_mle, sigma=np.sqrt(tau2_mle))
            else:
                return Normal(mu=mu0_mom, sigma=np.sqrt(tau0_sq_mom))
        else:
            raise ValueError(f"Unknown method '{method}'. Must be 'mle' or 'moments'.")

    def shrunk_posteriors(
            self,
            prior: Optional[Normal] = None,
            method: str = 'mle'
    ) -> NormalSeries:
        """
        Compute Empirical Bayes shrunk posterior Normal distributions for each category's mean.

        Pulls small-sample noisy group means toward the fitted grand mean prior:
            mu_j | D ~ Normal(mu_j_shrunk, sigma_j_shrunk)

        :param prior: Optional pre-specified Normal prior. If None, fitted via Empirical Bayes.
        :param method: 'mle' (default) or 'moments' for fitting prior.
        :return: NormalSeries containing posterior Normal distributions for all categories.
        """
        if prior is None:
            prior = self.empirical_bayes_prior(method=method)

        mu_0 = prior.mu
        tau0_sq = prior.sigma ** 2

        # Extract stats
        cats = self.keys()
        shrunk_normals = {}

        # Compute pooled variance for groups with n=1
        all_vars = [float(np.var(self._data[c].data, ddof=1)) for c in cats if len(self._data[c]) > 1]
        pooled_v = float(np.mean(all_vars)) if all_vars else float(prior.sigma ** 2)

        for cat in cats:
            r_dist: Ratio = self._data[cat]
            arr = np.asarray(r_dist.data, dtype=float)
            n = len(arr)
            y_bar = float(np.mean(arr)) if n > 0 else mu_0
            s2 = float(np.var(arr, ddof=1)) if n > 1 else pooled_v
            se2 = s2 / n if n > 0 else tau0_sq
            se2 = max(se2, 1e-8)

            # Shrinkage weight B_j = tau0^2 / (tau0^2 + se2)
            B_j = tau0_sq / (tau0_sq + se2)
            shrunk_mean = float(B_j * y_bar + (1.0 - B_j) * mu_0)
            post_var = float((tau0_sq * se2) / (tau0_sq + se2))
            post_std = max(np.sqrt(post_var), 1e-6)

            dist = Normal(mu=shrunk_mean, sigma=post_std)
            dist.with_label(cat)
            shrunk_normals[cat] = dist

        series_data = Series(shrunk_normals, name=f"shrunk_{self.name}" if self.name else "shrunk_posterior")
        series_data.index.name = self.index.name
        return NormalSeries(series_data)

    def shrinkage_summary(
            self,
            hdi: float = 0.95,
            prior: Optional[Normal] = None,
            method: str = 'mle'
    ) -> DataFrame:
        """
        Tidy DataFrame comparing raw category sample means with Empirical Bayes
        shrunk posterior estimates and HDIs.
        """
        if prior is None:
            prior = self.empirical_bayes_prior(method=method)

        shrunk_series = self.shrunk_posteriors(prior=prior)
        cats = self.keys()
        mu_0 = prior.mu
        tau0_sq = prior.sigma ** 2

        all_vars = [float(np.var(self._data[c].data, ddof=1)) for c in cats if len(self._data[c]) > 1]
        pooled_v = float(np.mean(all_vars)) if all_vars else float(prior.sigma ** 2)

        rows = []
        for cat in cats:
            r_dist: Ratio = self._data[cat]
            arr = np.asarray(r_dist.data, dtype=float)
            n = len(arr)
            y_bar = float(np.mean(arr)) if n > 0 else np.nan
            s2 = float(np.var(arr, ddof=1)) if n > 1 else pooled_v
            se = float(np.sqrt(s2 / n)) if n > 0 else np.nan
            se2 = s2 / n if n > 0 else tau0_sq

            shrunk_dist: Normal = shrunk_series[cat]
            low, high = shrunk_dist.interval(hdi)
            weight = float(tau0_sq / (tau0_sq + se2)) if n > 0 else 0.0

            rows.append({
                'category': cat,
                'n': n,
                'raw_mean': y_bar,
                'raw_se': se,
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
            prior: Optional[Normal] = None,
            method: str = 'mle',
            order_by: str = 'raw_mean',
            title: Optional[str] = None,
            ax: Optional[Axes] = None,
            **kwargs
    ) -> Axes:
        """
        Visualize Empirical Bayes shrinkage across continuous groups.

        Plots raw category means alongside shrunk posterior means and HDIs,
        with arrows indicating the directional shrinkage toward the prior grand mean.

        :param hdi: Highest Density Interval width (default 0.95).
        :param prior: Optional pre-fitted Normal prior.
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

        grand_mean = prior.mu

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
            label='Raw Sample Mean'
        )

        # Connect raw to shrunk with arrows
        for y, raw, shrunk in zip(y_positions, raw_means, shrunk_means):
            ax.annotate(
                '', xy=(shrunk, y), xytext=(raw, y),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2, alpha=0.7)
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels([f"{c} (n={summary_df.loc[c, 'n']})" for c in cats])
        ax.set_xlabel('Value')
        ax.set_title(
            title or f"Empirical Bayes Normal-Normal Shrinkage (Prior: N(μ={prior.mu:.2f}, σ={prior.sigma:.2f}))"
        )
        ax.legend(loc='best')
        return ax

    def histograms(
            self,
            bins: Union[int, float, Iterable[float]],
            min_pct: float = 0.0,
            max_pct: float = 1.0
    ):
        """
        Return a Series mapping distribution names to histograms.

        :param bins: int number of bins, float bin spacing, or sequence of bin
                     edges.
        :param min_pct: Lowest percentile of data to use for the histogram.
        :param max_pct: Highest percentile of data to use for the histogram.
        """
        dist: Ratio
        return Series({
            ix: dist.histogram(bins=bins, min_pct=min_pct, max_pct=max_pct)
            for ix, dist in self._data.items()
        })

    def unsplit(self) -> Ratio:
        """
        Return a Ratio with all the data for each Ratio in the
        RatioSeries concatenated.
        """
        return Ratio(
            data=concat([
                self._data[key].data for key in self.keys()
            ])
        )

    def _sample_series(self, unsplit_dist: Ratio, n_x: Series) -> 'RatioSeries':
        y_random = unsplit_dist.data.sample(frac=1)
        index = []
        for k in self.keys():
            index.extend([k] * n_x[k])
        y_random.index = index
        s_x = {
            k: Ratio(Series(data=y_random.loc[y_random.index == k]))
            for k in self.keys()
        }
        return RatioSeries(s_x)

    def plot_density_bars(
            self,
            bin_spacing: float,
            conditional: bool = True,
            width: float = 0.8,
            min_pct: float = 0.0,
            max_pct: float = 1.0,
            min_alpha: float = 0.0,
            color: Color = 'k',
            edge_color: Optional[Color] = 'grey',
            ax: Optional[Axes] = None
    ) -> Axes:
        """
        Plot a density bar for each Ratio distribution.

        :param bin_spacing: Spacing between each bin used to calculate
                            histogram of each Ratio distribution.
        :param conditional: Whether to set opacity relative to the highest bin
                            count of each Series (True) or all Series (False).
        :param width: Width of each bar,
        :param min_pct: Minimum quantile to plot from 0.0 to 1.0.
        :param max_pct: Maximum quantile to plot from 0.0 to 1.0.
        :param min_alpha: Alpha value for the opacity of bins with 0 count.
        :param color: Color for each density bar.
        :param edge_color: Color for edge of each bar. Set to None to omit
                           edges.
        :param ax: Optional matplotlib Axes instance.
        """
        ax = ax or new_axes()
        dist: Ratio
        # calculate histograms, counts and bin limits
        hists = [
            dist.histogram(
                bins=float(bin_spacing),
                min_pct=min_pct, max_pct=max_pct
            )
            for _, dist in self._data.items()
        ]
        lowest_bin = min([
            hist.index.get_level_values('min')[0] for hist in hists])
        highest_bin = max([
            hist.index.get_level_values('max')[-1] for hist in hists])
        max_counts = [hist.max() for hist in hists]
        max_max_count = max(max_counts)
        for x, hist, max_count in zip(
                range(len(self._data)), hists, max_counts
        ):
            max_count = hist.max()
            for (min_val, max_val), count in hist.items():
                # add segment interior
                if conditional is True:
                    alpha = clip(
                        min_alpha + (1 - min_alpha) * count / max_count,
                        min_alpha, 1.0
                    )
                else:
                    alpha = clip(
                        min_alpha + (1 - min_alpha) * count / max_max_count,
                        min_alpha, 1.0
                    )
                add_rectangle(
                    ax=ax,
                    width=width, height=max_val - min_val,
                    x_center=x, y_center=(min_val + max_val) / 2,
                    color=color, line_width=0,
                    alpha=float(alpha)
                )
            # add density edges
            if edge_color is not None:
                low_bin = hist.index.get_level_values('min')[0]
                high_bin = hist.index.get_level_values('max')[-1]
                add_rectangle(
                    ax=ax,
                    width=width, height=high_bin - low_bin,
                    x_center=x, y_center=(low_bin + high_bin) / 2,
                    fill=False, color=None, edge_color=edge_color
                )
        # format axes
        ax.set_xlim(-1, len(self._data))
        ax.set_ylim(lowest_bin - bin_spacing, highest_bin + bin_spacing)
        ax.set_xticks(range(len(self._data)))
        ax.set_xticklabels(self._data.index.to_list())

        format_axis_integer(ax.yaxis)
        ax.set_title(
            f'Distribution of '
            f'p({self.data.name}|{self.data.index.name})'
        )
        ax.set_xlabel(self.data.index.name)
        ax.set_ylabel(self.data.name)

        return ax

    def plot_density_distributions(
            self,
            ax: Optional[Axes] = None
    ) -> Axes:
        """
        Plot the distribution of probability density for each Ratio
        distribution.

        :param ax: Optional matplotlib Axes instance.
        """
        ax = ax or new_axes()
        name: str
        dist: Ratio
        for name, dist in self._data.items():
            kdeplot(
                x=dist.data,
                label=name, ax=ax
            )
        ax.legend()
        return ax
