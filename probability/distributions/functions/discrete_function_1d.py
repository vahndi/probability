from typing import Iterable, overload, Optional

from matplotlib.axes import Axes
from mpl_format.axes import AxesFormatter
from pandas import Series
from scipy.stats import rv_discrete

from probability.distributions.mixins.plottable_mixin import \
    DiscretePlottableMixin


class DiscreteFunction1d(object):

    def __init__(self,
                 distribution: rv_discrete,
                 method_name: str,
                 name: str,
                 parent: DiscretePlottableMixin):

        self._distribution = distribution
        self._method_name: str = method_name
        self._name: str = name
        self._method = getattr(distribution, method_name)
        self._parent: DiscretePlottableMixin = parent

    @overload
    def at(self, k: int) -> int:
        pass

    @overload
    def at(self, k: Iterable[int]) -> Series:
        pass

    def at(self, k):
        """
        Evaluation of the function for each value of k.
        """
        if isinstance(k, int) or isinstance(k, float):
            return self._method(k)
        elif isinstance(k, Iterable):
            return Series(index=k, data=self._method(k), name=self._name)

    def plot(self, k: Optional[Iterable[int]],
             color: str = 'C0',
             kind: str = 'bar',
             mean: bool = False,
             median: bool = False,
             std: bool = False,
             ax: Optional[Axes] = None,
             **kwargs) -> Axes:
        """
        Plot the function.

        :param k: Range of values of k to plot p(k) over.
        :param color: Optional color for the series.
        :param kind: Kind of plot e.g. 'bar', 'line'.
        :param mean: Whether to show marker and label for the mean.
        :param median: Whether to show marker and label for the median.
        :param std: Whether to show marker and label for the standard deviation.
        :param ax: Optional matplotlib axes to plot on.
        :param kwargs: Additional arguments for the matplotlib plot function.
        """
        if k is None:
            if (
                    hasattr(self._parent, 'lower_bound') and
                    hasattr(self._parent, 'upper_bound')
            ):
                k = range(self._parent.lower_bound,
                          self._parent.upper_bound + 1)
            else:
                raise ValueError('Must pass k if distribution has no bounds.')

        data: Series = self.at(k)
        axf = AxesFormatter(axes=ax)
        ax = axf.axes

        # special kwargs
        vlines = None
        if 'vlines' in kwargs.keys():
            vlines = kwargs.pop('vlines')
        if 'label' not in kwargs.keys():
            kwargs['label'] = self._parent.label

        if self._method_name == 'pmf':
            data.plot(kind=kind, color=color, ax=axf.axes, **kwargs)
        elif self._method_name == 'cdf':
            data.plot(kind='line', color=color, drawstyle='steps-post',
                      ax=axf.axes, **kwargs)
        else:
            raise ValueError('plot not implemented for {}'.format(self._name))
        if vlines:
            axf.axes.vlines(x=k, ymin=0, ymax=data.values, color=color)

        y_min = axf.get_y_min()
        y_max = axf.get_y_max()
        x_mean = self._distribution.mean()
        if mean:
            axf.add_v_lines(x=x_mean, y_min=y_min, y_max=y_max,
                            line_styles='--', colors=color)
            axf.add_text(x=x_mean, y=self._distribution.pmf(x_mean),
                         text=f'mean={x_mean: 0.3f}', color=color,
                         ha='center', va='bottom')
        if median:
            x_median = self._distribution.median()
            axf.add_v_lines(x=x_median, y_min=y_min, y_max=y_max,
                            line_styles='-.', colors=color)
            axf.add_text(x=x_median, y=self._distribution.pmf(x_median),
                         text=f'median={x_median: 0.3f}', color=color,
                         ha='center', va='bottom')
        if std:
            x_std = self._distribution.std()
            axf.add_v_lines(x=[x_mean - x_std, x_mean + x_std],
                            y_min=y_min, y_max=y_max,
                            line_styles=':', colors=color)
            axf.add_text(x=x_mean - x_std / 2,
                         y=self._distribution.pmf(x_mean - x_std / 2),
                         text=f'std={x_std: 0.3f}', color=color,
                         ha='center', va='bottom')

        ax.set_xlabel(self._parent.x_label)

        if self._parent.y_label:
            ax.set_ylabel(self._parent.y_label)
        else:
            if self._method_name == 'pmf':
                ax.set_ylabel('P(K = k)')
            elif self._method_name == 'cdf':
                ax.set_ylabel('P(K ≤ k)')
            else:
                ax.set_ylabel(self._name)

        return ax
