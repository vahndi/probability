from math import floor, ceil

from matplotlib.axes import Axes
from numpy import linspace
from pandas import DataFrame, Series
from scipy.stats import rv_continuous
from typing import Iterable, Optional, Tuple, Union, Dict, Any

from probability.plotting import Color, add_v_density, new_axes


from compound_types.built_ins import FloatIterable
from probability.distributions.mixins.plottable_mixin import \
    ContinuousPlottableMixin
from probability.distributions.mixins.rv_mixins import RVS1dMixin, \
    Moment1dMixin, EntropyMixin, Median1dMixin, Mean1dMixin, StD1dMixin, \
    Var1dMixin, Interval1dMixin, Support1dMixin, PDF1dMixin, \
    CDFContinuous1dMixin, SFContinuous1dMixin, PPFContinuous1dMixin, \
    ISFContinuous1dMixin, StatMixin
from probability.distributions.mixins.rv_series import RVContinuousSeries
from probability.distributions.protocols import (
    SupportsContinuousRV1d, SupportsRVS1d
)
from probability.utils import is_scalar


class RVContinuous1dMixin(
    RVS1dMixin, Moment1dMixin, EntropyMixin, Median1dMixin, Mean1dMixin,
    StD1dMixin, Var1dMixin, Interval1dMixin, Support1dMixin, PDF1dMixin,
    CDFContinuous1dMixin, SFContinuous1dMixin, PPFContinuous1dMixin,
    ISFContinuous1dMixin, ContinuousPlottableMixin, StatMixin, object
):

    _distribution: Union[rv_continuous, RVContinuousSeries]
    lower_bound: float
    upper_bound: float

    @staticmethod
    def fit(data: FloatIterable) -> 'RVContinuous1dMixin':
        """
        Fit a  distribution to the data from a single experiment.

        :param data: Iterable of data to fit to.
        """
        raise NotImplementedError

    def plot(self: SupportsContinuousRV1d,
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
             ax: Axes = None, **kwargs) -> Axes:
        """
        Plot the PDF of the distribution.

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
        return self.pdf().plot(
            x=x, kind=kind, color=color,
            mean=mean, median=median, mode=mode, std=std,
            ci=ci, fill_interval=fill_interval, fill_alpha=fill_alpha,
            ax=ax, **kwargs
        )


    @staticmethod
    def plot_densities(
            data: Optional[DataFrame] = None,
            labels: Union[Series, str] = None,
            distributions: Union[Series, str] = None,
            color: Color = 'k',
            color_min: Optional[Color] = None,
            width: float = 0.8,
            num_strips: int = 100,
            ax: Optional[Axes] = None
    ) -> Axes:
        """
        Plot a density plot (continuous boxplot) of distribution pdfs.

        :param data: Optional DataFrame containing labels and distributions.
        :param labels: Series of labels or name of column.
        :param distributions: Series of distributions or name of column.
        :param num_strips: Number of strips for each density bar.
        :param color: The color of the density bar.
        :param color_min: Optional 2nd color to fade out to.
        :param width: The bar width.
        :param ax: Optional matplotlib Axes instance.
        """
        # check arguments
        ax = ax or new_axes()
        if data is None:
            if not (isinstance(labels, Series) and
                    isinstance(distributions, Series)):
                raise TypeError(
                    'If data is not given, '
                    'labels and distributions must both be Series'
                )
        else:
            if not isinstance(data, DataFrame):
                raise TypeError('data must be a DataFrame')
            if not (isinstance(labels, str) and isinstance(distributions, str)):
                raise TypeError(
                    'If data is given, '
                    'labels and distributions must both be str'
                )
            labels: Series = data[labels]
            distributions: Series = data[distributions]
        # plot densities
        distribution: SupportsContinuousRV1d
        y_to_z: Dict[Any, Series] = {}
        max_z = 0
        min_y = 1e6
        max_y = -1e6
        for x_label, distribution in zip(labels, distributions):
            y_dist = linspace(distribution.lower_bound,
                              distribution.upper_bound,
                              num_strips + 1)
            min_y = min(min_y, y_dist[0])
            max_y = max(max_y, y_dist[-1])
            y_to_z[x_label] = Series(
                index=y_dist, data=distribution.pdf().at(y_dist)
            )
            max_z = max(max_z, y_to_z[x_label].max())
        label_ix = {label: i + 1 for i, label in enumerate(list(labels))}
        for label, z_values in y_to_z.items():
            add_v_density(
                ax=ax,
                x=label_ix[label], y_to_z=z_values,
                color=color, color_min=color_min,
                width=width, z_max=max_z
            )
        num_labels = len(labels)
        ax.set_xticks(range(1, num_labels + 1))
        ax.set_xticklabels(labels)
        ax.set_xlim(0, num_labels + 1)
        ax.set_ylim(floor(min_y), ceil(max_y))
        ax.set_ylabel('x')
        return ax

    def __le__(self, other: Union[SupportsRVS1d, int, float]) -> float:

        if is_scalar(other):
            return self.cdf().at(other)
        elif isinstance(other, (RVS1dMixin, SupportsRVS1d)):
            return super(RVContinuous1dMixin, self).__le__(other)
        else:
            raise TypeError('other must be of type int, float or Rvs1dMixin')

    def __lt__(self, other: Union[SupportsRVS1d, int, float]) -> float:

        return self.__le__(other)

    def __ge__(self, other: Union[SupportsRVS1d, int, float]) -> float:

        if is_scalar(other):
            return 1 - self.__le__(other)
        elif isinstance(other, (RVS1dMixin, SupportsRVS1d)):
            return super(RVContinuous1dMixin, self).__ge__(other)
        else:
            raise TypeError('other must be of type int, float or Rvs1dMixin')

    def __gt__(self, other: Union[SupportsRVS1d, int, float]) -> float:

        return self.__ge__(other)

