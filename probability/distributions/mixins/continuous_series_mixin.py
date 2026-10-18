from typing import TypeVar, Union, Mapping, List, Optional, Any

from matplotlib.axes import Axes
from numpy import linspace, arange, inf
from pandas import Series

from probability.distributions.functions.continuous_function_1d_series import \
    ContinuousFunction1dSeries
from probability.distributions.protocols import SupportsContinuousRV1d
from probability.models.utils import loop_variable
from probability.plotting import (
    Color,
    add_h_density,
    add_rectangle,
    add_v_density,
    cross_fade,
    new_axes,
)

CSM = TypeVar('CSM', bound='ContinuousSeriesMixin')


class ContinuousSeriesMixin(object):

    _data: Series

    @property
    def name(self):
        return self._data.name

    @property
    def data(self) -> Series:
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
            resolution: Optional[int] = None,
            edges: bool = False,
            orient: str = 'v',
            ax: Optional[Axes] = None
    ) -> Axes:
        """
        Plot each distribution as a density bar.

        :param color: Color of each bar, all bars or list to cycle through.
        :param color_min: Min color of each bar, all bars or list to cycle
                          through.
        :param width: Width of each bar.
        :param hdi: Highest Density Interval width for each distribution.
        :param z_max: Optional normalizing constant to divide each bar's height
                      by.
        :param resolution: Number of density elements per unit y. If left as
                           None, 100 total elements will be used.
        :param edges: Whether to plot the edges of each bar.
        :param orient: Orientation. One of {'v', 'h'}.
        :param ax: Optional matplotlib Axes instance.
        """
        ax = ax or new_axes()
        dist: SupportsContinuousRV1d
        num_dists = len(self._data)
        color = loop_variable(color, num_dists)
        color_min = loop_variable(color_min, num_dists)
        width = loop_variable(width, num_dists)
        z_max = loop_variable(z_max, num_dists)
        lowest, highest = inf, -inf
        for i, (ix, dist) in enumerate(self._data.items()):
            lower, upper = dist.hdi(hdi)

            lowest = min(lowest, lower)
            highest = max(highest, upper)
            n_bars = (
                round(resolution * (upper - lower))
                if resolution is not None else 100
            )
            y_to_z = dist.pdf().at(linspace(lower, upper, n_bars + 1))
            if orient == 'h':
                add_h_density(
                    ax=ax,
                    y=i + 1,
                    x_to_z=y_to_z,
                    color=color[i], color_min=color_min[i],
                    height=width[i],
                    z_max=z_max[i],
                    v_align='center'
                )
                if edges:
                    add_rectangle(
                        ax=ax,
                        width=y_to_z.index[-1] - y_to_z.index[0],
                        height=width[i],
                        x_left=y_to_z.index[0],
                        y_bottom=i + 1 - width[i] / 2,
                        edge_color=(
                            color[i] if color_min[i] is None else
                            cross_fade(color_min[i], color[i], 0.5)
                        ),
                        fill=False
                    )
            else:
                add_v_density(
                    ax=ax,
                    x=i + 1,
                    y_to_z=y_to_z,
                    color=color[i], color_min=color_min[i],
                    width=width[i],
                    z_max=z_max[i],
                    h_align='center'
                )
                if edges:
                    add_rectangle(
                        ax=ax,
                        width=width[i],
                        height=y_to_z.index[-1] - y_to_z.index[0],
                        x_left=i + 1 - width[i] / 2,
                        y_bottom=y_to_z.index[0],
                        edge_color=(
                            color[i] if color_min[i] is None else
                            cross_fade(color_min[i], color[i], 0.5)
                        ),
                        fill=False
                    )
        if orient == 'v':
            ax.set_xlim(0, num_dists + 1)
            ax.set_xticks(range(1, num_dists + 1))
            ax.set_xticklabels(self._data.index)
            ax.set_ylim(lowest - (highest - lowest) / 20,
                        highest + (highest - lowest) / 20)
            ax.set_ylabel(str(self.name))
        else:
            ax.set_ylim(0, num_dists + 1)
            ax.set_yticks(range(1, num_dists + 1))
            ax.set_yticklabels(self._data.index)
            ax.set_xlim(lowest - (highest - lowest) / 20,
                        highest + (highest - lowest) / 20)
            ax.set_xlabel(str(self.name))
        return ax

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def keys(self) -> List[Any]:
        return list(self._data.index)

    def __repr__(self):

        return repr(self._data)
