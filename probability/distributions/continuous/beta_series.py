from typing import Optional, Mapping, Union, List

from mpl_format.axes import AxesFormatter
from mpl_format.compound_types import Color
from mpl_format.utils.color_utils import cross_fade
from numpy import linspace
from numpy.ma import arange
from pandas import Series

from probability.distributions import Beta
from probability.distributions.functions.continuous_function_1d_series import \
    ContinuousFunction1dSeries
from probability.models.utils import loop_variable


class BetaSeries(object):

    def __init__(self, data: Series):
        """
        Create a new BetaSeries.

        :param data: Series of Beta distributions.
        """
        self._data: Series = data

    @property
    def data(self) -> Union[Series, Mapping[str, Beta]]:
        return self._data

    def means(self) -> Union[Series, Mapping[str, float]]:
        return self._data.map(lambda d: d.mean())

    # def pdfs(self, at: Optional[FloatArray1d] = None) -> DataFrame:
    #
    #     if at is None:
    #         at = linspace(0, 1, 101)
    #     beta: Beta
    #     return DataFrame(
    #         data=[
    #             beta.pdf().at(at) for beta in self._data.values
    #         ],
    #         index=self._data.index,
    #         columns=at
    #     ).T

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
            color: Union[Color, List[Color]],
            color_min: Optional[Union[Color, List[Color]]] = None,
            width: Union[float, List[float]] = 0.8,
            min_pct: float = 0.025, max_pct: float = 0.975,
            z_max: Optional[Union[float, List[float]]] = None,
            edges: bool = False,
            axf: Optional[AxesFormatter] = None
    ) -> AxesFormatter:
        """
        Plot each distribution as a density bar.

        :param color: Color of each bar, all bars or list to cycle through.
        :param color_min: Min color of each bar, all bars or list to cycle
                          through.
        :param width: Width of each bar.
        :param min_pct: Min ppf to start plotting at for each distribution.
        :param max_pct: Max ppf to end plotting at for each distribution.
        :param z_max: Optional normalizing constant to divide each bar's height
                      by.
        :param edges: Whether to plot the edges of each bar.
        :param axf: Optional AxesFormatter instance.
        """
        axf = axf or AxesFormatter()
        beta: Beta
        num_dists = len(self._data)
        color = loop_variable(color, num_dists)
        color_min = loop_variable(color_min, num_dists)
        width = loop_variable(width, num_dists)
        z_max = loop_variable(z_max, num_dists)

        for i, (ix, beta) in enumerate(self._data.items()):
            y_to_z = beta.pdf().at(linspace(
                    beta.ppf().at(min_pct), beta.ppf().at(max_pct), 96
            ))
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
        axf.set_x_lim(0, num_dists + 1)
        axf.x_ticks.set_locations(range(1, num_dists + 1))
        axf.x_ticks.set_labels(self._data.index)
        axf.y_ticks.set_locations(arange(0, 1.1, 0.1))
        axf.set_y_lim(-0.05, 1.05)
        return axf
