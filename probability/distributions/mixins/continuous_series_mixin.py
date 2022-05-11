from typing import TypeVar, Union, Mapping, List, Optional

from numpy import linspace, arange
from pandas import Series

from mpl_format.axes import AxesFormatter
from mpl_format.compound_types import Color
from mpl_format.utils.color_utils import cross_fade
from probability.distributions.functions.continuous_function_1d_series import \
    ContinuousFunction1dSeries
from probability.distributions.mixins.rv_mixins import PPFContinuous1dMixin, \
    PDF1dMixin
from probability.models.utils import loop_variable

CSM = TypeVar('CSM', bound='ContinuousSeriesMixin')


class ContinuousSeriesMixin(object):

    _data: Series

    def means(self) -> Union[Series, Mapping[str, float]]:
        return self._data.map(lambda d: d.mean())

    def pdfs(self) -> ContinuousFunction1dSeries:

        return ContinuousFunction1dSeries(
            self._data.map(lambda d: d.pdf())
        )

    def ppfs(self) -> ContinuousFunction1dSeries:

        return ContinuousFunction1dSeries(
            self._data.map(lambda d: d.ppf())
        )

    def plot_density_bars(
            self,
            color: Union[Color, List[Color]] = 'k',
            color_min: Optional[Union[Color, List[Color]]] = None,
            width: Union[float, List[float]] = 0.8,
            hdi: float = 0.95,
            z_max: Optional[Union[float, List[float]]] = None,
            resolution: int = 100,
            edges: bool = False,
            orient: str = 'v',
            axf: Optional[AxesFormatter] = None
    ) -> AxesFormatter:
        """
        Plot each distribution as a density bar.

        :param color: Color of each bar, all bars or list to cycle through.
        :param color_min: Min color of each bar, all bars or list to cycle
                          through.
        :param width: Width of each bar.
        :param hdi: Highest Density Interval width for each distribution.
        :param z_max: Optional normalizing constant to divide each bar's height
                      by.
        :param resolution: Number of density elements per unit y.
        :param edges: Whether to plot the edges of each bar.
        :param orient: Orientation. One of {'v', 'h'}.
        :param axf: Optional AxesFormatter instance.
        """
        axf = axf or AxesFormatter()
        dist: Union[PPFContinuous1dMixin, PDF1dMixin]
        num_dists = len(self._data)
        color = loop_variable(color, num_dists)
        color_min = loop_variable(color_min, num_dists)
        width = loop_variable(width, num_dists)
        z_max = loop_variable(z_max, num_dists)

        for i, (ix, dist) in enumerate(self._data.items()):
            lower, upper = dist.hdi(hdi)
            y_to_z = dist.pdf().at(linspace(lower, upper, resolution + 1))
            if orient == 'h':
                axf.add_h_density(
                    y=i + 1,
                    x_to_z=y_to_z,
                    color=color[i], color_min=color_min[i],
                    height=width[i],
                    z_max=z_max[i],
                    v_align='center'
                )
                if edges:
                    axf.add_rectangle(
                        width=y_to_z.index[-1] - y_to_z.index[0],
                        height=width[i],
                        x_left=y_to_z.index[0],
                        y_bottom=i + 1 - width[i] / 2,
                        edge_color=(
                            color if color_min is None else
                            cross_fade(color_min[i], color[i], 0.5)
                        ),
                        fill=False
                    )
            else:
                axf.add_v_density(
                    x=i + 1,
                    y_to_z=y_to_z,
                    color=color[i], color_min=color_min[i],
                    width=width[i],
                    z_max=z_max[i],
                    h_align='center'
                )
                if edges:
                    axf.add_rectangle(
                        width=width[i],
                        height=y_to_z.index[-1] - y_to_z.index[0],
                        x_left=i + 1 - width[i] / 2,
                        y_bottom=y_to_z.index[0],
                        edge_color=(
                            color if color_min is None else
                            cross_fade(color_min[i], color[i], 0.5)
                        ),
                        fill=False
                    )
        if orient == 'v':
            axf.set_x_lim(0, num_dists + 1)
            axf.x_ticks.set_locations(range(1, num_dists + 1))
            axf.x_ticks.set_labels(self._data.index)
            axf.y_ticks.set_locations(arange(0, 1.1, 0.1))
            axf.set_y_lim(-0.05, 1.05)
        else:
            axf.set_y_lim(0, num_dists + 1)
            axf.y_ticks.set_locations(range(1, num_dists + 1))
            axf.y_ticks.set_labels(self._data.index)
            axf.x_ticks.set_locations(arange(0, 1.1, 0.1))
            axf.set_x_lim(-0.05, 1.05)
        return axf

    def __repr__(self):

        return repr(self._data)