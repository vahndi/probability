from typing import Iterable, overload, Optional, Tuple, Union


from matplotlib.axes import Axes
from numpy import linspace
from pandas import Series
from scipy.stats import rv_continuous

from probability.plotting import new_axes

from probability.distributions.mixins.plottable_mixin import \
    ContinuousPlottableMixin
from probability.distributions.mixins.rv_series import RVContinuousSeries


class ContinuousFunction1d(object):

    def __init__(self,
                 distribution: Union[rv_continuous, RVContinuousSeries],
                 method_name: str,
                 name: str,
                 parent: ContinuousPlottableMixin):

        self._distribution = distribution
        self._method_name: str = method_name
        self._name: str = name
        self._method = getattr(distribution, method_name)
        self._parent: ContinuousPlottableMixin = parent

    @overload
    def at(self, x: float) -> float:
        pass

    @overload
    def at(self, x: Iterable) -> Series:
        pass

    def at(self, x):
        """
        Log of the probability density function of the given RV.
        """
        if isinstance(x, float) or isinstance(x, int):
            return self._method(x)
        elif isinstance(x, Iterable):
            return Series(index=x, data=self._method(x), name=self._name)

    def plot(self,
             x: Optional[Iterable] = None,
             kind: str = 'line',
             color: str = 'C0',
             mean: bool = False,
             median: bool = False,
             mode: bool = False,
             std: bool = False,
             ci: Optional[Union[float, Iterable[float]]] = None,
             fill_interval: Optional[
                 Tuple[Optional[float], Optional[float]]
             ] = None,
             fill_alpha: float = 0.3,
             ax: Optional[Axes] = None,
             **kwargs) -> Axes:
        """
        Plot the function.

        :param x: Range of values of x to plot p(x) over.
        :param kind: Kind of plot e.g. 'bar', 'line'.
        :param color: Optional color for the series.
        :param mean: Whether to show marker and label for the mean.
        :param median: Whether to show marker and label for the median.
        :param mode: Whether to show marker and label for the mode.
        :param std: Whether to show marker and label for the standard deviation.
        :param ci: Optional confidence level (float in (0, 1)) or iterable of
                   confidence levels to shade under the curve.
        :param fill_interval: Optional (low, high) interval to shade under the
                              curve. Use None for open lower/upper tail bounds.
        :param fill_alpha: Alpha transparency for shaded regions (default 0.3).
        :param ax: Optional matplotlib axes to plot on.
        :param kwargs: Additional arguments for the matplotlib plot function.
        """
        if x is None:
            if (
                    hasattr(self._parent, 'lower_bound') and
                    hasattr(self._parent, 'upper_bound')
            ):
                x = linspace(self._parent.lower_bound,
                             self._parent.upper_bound, 1001)
            else:
                raise ValueError('Must pass x if distribution has no bounds.')

        data: Series = self.at(x)
        ax = ax or new_axes()

        if self._method_name in ('pdf', 'cdf', 'logpdf'):
            if 'label' not in kwargs.keys():
                kwargs['label'] = self._parent.label
            data.plot(kind=kind, color=color, ax=ax, **kwargs)
        else:
            raise ValueError('plot not implemented for {}'.format(self._name))

        # shaded confidence intervals
        if ci is not None:
            ci_levels = [ci] if isinstance(ci, (int, float)) else list(ci)
            for level in ci_levels:
                low, high = self._distribution.interval(level)
                sub = data.loc[(data.index >= low) & (data.index <= high)]
                ax.fill_between(
                    sub.index.to_numpy(dtype=float),
                    0,
                    sub.to_numpy(dtype=float),
                    color=color,
                    alpha=fill_alpha,
                    label=f'{level:.1%} CI'
                )

        # shaded custom interval / tail
        if fill_interval is not None:
            low, high = fill_interval
            mask = Series(True, index=data.index)
            if low is not None:
                mask &= (data.index >= low)
            if high is not None:
                mask &= (data.index <= high)
            sub = data.loc[mask]
            ax.fill_between(
                sub.index.to_numpy(dtype=float),
                0,
                sub.to_numpy(dtype=float),
                color=color,
                alpha=fill_alpha
            )

        # stats
        y_min, y_max = ax.get_ylim()
        x_mean = self._distribution.mean()
        if mean:
            ax.vlines(x=x_mean, ymin=y_min, ymax=y_max,
                      linestyles='--', colors=color)
            ax.text(x=x_mean, y=self._distribution.pdf(x_mean),
                    s=f'mean={x_mean: 0.3f}', color=color,
                    ha='center', va='bottom')
        if median:
            x_median = self._distribution.median()
            ax.vlines(x=x_median, ymin=y_min, ymax=y_max,
                      linestyles='-.', colors=color)
            ax.text(x=x_median, y=self._distribution.pdf(x_median),
                    s=f'median={x_median: 0.3f}', color=color,
                    ha='center', va='bottom')
        if mode:
            x_mode = self._parent.mode()
            ax.vlines(x=x_mode, ymin=y_min, ymax=y_max,
                      linestyles='-.', colors=color)
            ax.text(x=x_mode, y=self._distribution.pdf(x_mode),
                    s=f'mode={x_mode: 0.3f}', color=color,
                    ha='center', va='bottom')
        if std:
            x_std = self._distribution.std()
            ax.vlines(x=[x_mean - x_std, x_mean + x_std],
                      ymin=y_min, ymax=y_max,
                      linestyles=':', colors=color)
            ax.text(x=x_mean - x_std / 2,
                    y=self._distribution.pdf(x_mean - x_std / 2),
                    s=f'std={x_std: 0.3f}', color=color,
                    ha='center', va='bottom')

        ax.set_xlabel(self._parent.x_label)

        if self._parent.y_label:
            ax.set_ylabel(self._parent.y_label)
        else:
            if self._method_name == 'pdf':
                ax.set_ylabel('P(X = x)')
            elif self._method_name == 'cdf':
                ax.set_ylabel('P(X ≤ x)')
            elif self._method_name == 'logpdf':
                ax.set_ylabel('log P(X = x)')
            else:
                ax.set_ylabel(self._name)

        return ax


    def __str__(self):

        return f'{self._name}[{self._parent}]'
