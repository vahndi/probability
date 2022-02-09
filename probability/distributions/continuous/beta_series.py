from typing import Optional, Mapping, Union, List

from numpy import linspace
from numpy.ma import arange
from pandas import Series, DataFrame

from mpl_format.axes import AxesFormatter
from mpl_format.compound_types import Color
from mpl_format.utils.color_utils import cross_fade
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

    @staticmethod
    def from_bool_frame(
            data: DataFrame,
            prior_alpha: float = 0,
            prior_beta: float = 0,
            name: str = ''
    ):
        """
        Create a new BetaSeries using the counts of True and False or 1 and 0
        in a DataFrame.

        :param data: Data with True / False counts.
        :param prior_alpha: Value for alpha assuming these represent posterior
                            distributions.
        :param prior_beta: Value for alpha assuming these represent posterior
                            distributions.
        :param name: Name for the Series.
        """
        betas = {}
        for col in data.columns:
            betas[col] = Beta(
                alpha=prior_alpha + (data[col] == 1).sum(),
                beta=prior_beta + (data[col] == 0).sum()
            )
        betas = Series(data=betas, name=name)
        return BetaSeries(betas)

    @staticmethod
    def from_proportions(data: DataFrame):
        """
        Fit to a DataFrame of proportions. Returns a Series with one item for
        each column in data.
        """
        return BetaSeries(Series({
            column: Beta.fit(data[column].dropna())
            for column in data.columns
        }))

    @property
    def data(self) -> Union[Series, Mapping[str, Beta]]:
        return self._data

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
        beta: Beta
        num_dists = len(self._data)
        color = loop_variable(color, num_dists)
        color_min = loop_variable(color_min, num_dists)
        width = loop_variable(width, num_dists)
        z_max = loop_variable(z_max, num_dists)

        for i, (ix, beta) in enumerate(self._data.items()):
            lower, upper = beta.hdi(hdi)
            y_to_z = beta.pdf().at(linspace(lower, upper, resolution + 1))
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
