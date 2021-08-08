from math import floor, ceil

from matplotlib.axes import Axes
from mpl_format.axes import AxesFormatter
from mpl_format.axes.axis_utils import new_axes
from mpl_format.compound_types import Color
from numpy import linspace
from pandas import DataFrame, Series
from scipy.stats import rv_continuous
from typing import Iterable, Optional, Union, Dict, Any

from compound_types.built_ins import FloatIterable
from probability.distributions.mixins.plottable_mixin import \
    ContinuousPlottableMixin
from probability.distributions.mixins.rv_mixins import RVS1dMixin, \
    Moment1dMixin, EntropyMixin, Median1dMixin, Mean1dMixin, StD1dMixin, \
    Var1dMixin, Interval1dMixin, Support1dMixin, PDF1dMixin, \
    CDFContinuous1dMixin, SFContinuous1dMixin, PPFContinuous1dMixin, \
    ISFContinuous1dMixin, StatMixin
from probability.distributions.mixins.rv_series import RVContinuousSeries


class RVContinuous1dMixin(
    RVS1dMixin, Moment1dMixin, EntropyMixin, Median1dMixin, Mean1dMixin,
    StD1dMixin, Var1dMixin, Interval1dMixin, Support1dMixin, PDF1dMixin,
    CDFContinuous1dMixin, SFContinuous1dMixin, PPFContinuous1dMixin,
    ISFContinuous1dMixin, ContinuousPlottableMixin, StatMixin, object
):

    _distribution: Union[rv_continuous, RVContinuousSeries]

    @property
    def lower_bound(self) -> float:
        raise NotImplementedError

    @property
    def upper_bound(self) -> float:
        raise NotImplementedError

    @staticmethod
    def fit(data: FloatIterable) -> 'RVContinuous1dMixin':
        """
        Fit a  distribution to the data from a single experiment.

        :param data: Iterable of data to fit to.
        """
        raise NotImplementedError

    def plot(self,
             x: Optional[Iterable] = None,
             kind: str = 'line',
             color: str = 'C0',
             ax: Axes = None, **kwargs) -> Axes:
        """
        Plot the PDF of the distribution.

        :param x: Range of values of x to plot p(x) over.
        :param kind: Kind of plot e.g. 'bar', 'line'.
        :param color: Optional color for the series.
        :param ax: Optional matplotlib axes to plot on.
        :param kwargs: Additional arguments for the matplotlib plot function.
        """
        return self.pdf().plot(x=x, kind=kind, color=color, ax=ax, **kwargs)

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
        axf = AxesFormatter(ax)
        distribution: RVContinuous1dMixin
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
            axf.add_v_density(
                x=label_ix[label], y_to_z=z_values,
                color=color, color_min=color_min,
                width=width, z_max=max_z
            )
        num_labels = len(labels)
        axf.x_axis.axis.set_ticks(range(1, num_labels + 1))
        axf.x_axis.axis.set_ticklabels(labels)
        axf.set_x_lim(0, num_labels + 1)
        axf.set_y_lim(floor(min_y), ceil(max_y))
        axf.set_text(y_label='x')
        return axf.axes

    def __le__(self, other: Union['RVS1dMixin', int, float]) -> float:

        if type(other) in (int, float):
            return self.cdf().at(other)
        elif isinstance(other, RVS1dMixin):
            return super(RVContinuous1dMixin, self).__le__(other)
        else:
            raise TypeError('other must be of type int, float or Rvs1dMixin')

    def __lt__(self, other: Union['RVS1dMixin', int, float]) -> float:

        return self.__le__(other)

    def __ge__(self, other: Union['RVS1dMixin', int, float]) -> float:

        if type(other) in (int, float):
            return 1 - self.__le__(other)
        elif isinstance(other, RVS1dMixin):
            return super(RVContinuous1dMixin, self).__ge__(other)
        else:
            raise TypeError('other must be of type int, float or Rvs1dMixin')

    def __gt__(self, other: Union['RVS1dMixin', int, float]) -> float:

        return self.__ge__(other)
