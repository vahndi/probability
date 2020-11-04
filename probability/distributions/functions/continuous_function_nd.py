from typing import overload, Iterable, Union, Optional, List

from matplotlib.axes import Axes
from matplotlib.tri import UniformTriRefiner, Triangulation
from mpl_format.axes.axis_utils import new_axes
from numpy import array, clip, dstack, meshgrid, ndarray
from numpy.linalg import norm
from numpy.ma import clip
from pandas import Series, MultiIndex
from scipy.stats import rv_continuous

from probability.distributions.mixins.plottable_mixin import \
    ContinuousPlottableNdMixin


class ContinuousFunctionNd(object):

    def __init__(self,
                 distribution: rv_continuous,
                 method_name: str,
                 name: str,
                 num_dims: int,
                 parent: ContinuousPlottableNdMixin):
        """
        :param distribution: The scipy distribution to calculate with.
        :param method_name: The name of the method to call on the distribution.
        :param name: An intuitive name for the function.
        :param num_dims: The number of dimensions, K, of the function.
        :param parent: The parent distribution object, used to call str(...) for
                       series labels.
        """
        self._distribution = distribution
        self._num_dims = num_dims
        self._method_name: str = method_name
        self._name: str = name
        self._method = getattr(distribution, method_name)
        self._parent: ContinuousPlottableNdMixin = parent

    @overload
    def at(self, x: Iterable[float]) -> float:
        pass

    @overload
    def at(self, x: Iterable[Iterable]) -> Series:
        pass

    @overload
    def at(self, x: ndarray) -> Series:
        pass

    def at(self, x):
        """
        Evaluate the function for each value of [x1, x2, ..., xk] given as x.

        :param x: [x1, x2, ..., xk] or [[x11, x12, ..., x1k],
                                        [x21, x22, ..., x2k],
                                        ...]
        """
        x = array(x)
        if x.ndim == 1:
            return self._method(x)
        elif x.ndim == 2:
            return Series(
                index=MultiIndex.from_arrays(
                    arrays=x.T,
                    names=[f'x{num}' for num in range(1, self._num_dims + 1)]
                ), data=self._method(x), name=f'{self._name}({self._parent})'
            )

    def plot(
            self,
            x: Iterable,
            kind: str = 'line',
            colors: Optional[List[str]] = None,
            ax: Optional[Axes] = None,
            **kwargs
    ) -> Axes:
        """
        Plot the marginal distribution of each component.

        :param x: Range of values of x to plot p(x) over.
        :param kind: Kind of plot e.g. 'bar', 'line'.
        :param colors: Optional list of colors for each series.
        :param ax: Optional matplotlib axes to plot on.
        :param kwargs: Additional arguments for the matplotlib plot function.
        """
        parent = self._parent
        if colors is None:
            colors = [f'C{i}' for i in range(len(parent.names))]
        if len(colors) != len(parent.names):
            raise ValueError(f'Pass 0 or {len(parent.names)} colors.')
        ax = ax or new_axes()
        for k, color in zip(parent.names, colors):
            data = getattr(parent[k], self._method_name)().at(x)
            data.plot(x=x, kind=kind, color=color,
                      ax=ax, label=f'{k}', **kwargs)
        ax.legend()
        ax.set_xlabel(parent.x_label)
        ax.set_ylabel(f'{self._name}({parent.x_label})')

        return ax

    def plot_2d(self,
                x1: Union[Iterable, ndarray],
                x2: Union[Iterable, ndarray],
                color_map: str = 'viridis', ax: Optional[Axes] = None,
                **kwargs) -> Axes:
        """
        Plot a 2-dimensional function as a grid heat-map.

        N.B. don't use for distributions where calculating the function for the
        full range of x1 and x2 values would cause an error e.g. for a Dirichlet
        where x1 + x2 must equal 1.

        :param x1: Range of values of x1 to plot p(x1, x2) over.
        :param x2: Range of values of x2 to plot p(x1, x2) over.
        :param color_map: Optional colormap for the heat-map.
        :param ax: Optional matplotlib axes to plot on.
        :param kwargs: Additional arguments for contourf method.
        """
        x1_grid, x2_grid = meshgrid(x1, x2)
        x1_x2 = dstack((x1_grid, x2_grid))
        f = self._method(x1_x2)
        ax = ax or new_axes()
        ax.contourf(x1_grid, x2_grid, f, cmap=color_map, **kwargs)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        return ax

    def plot_simplex(self, num_contours: int = 100, num_sub_div: int = 8,
                     color_map: str = 'viridis', border: bool = True,
                     ax: Optional[Axes] = None,
                     **kwargs) -> Axes:
        """
        Plot a 3-dimensional functions as a simplex heat-map.

        :param num_contours: The number of levels of contours to plot.
        :param num_sub_div: Number of recursive subdivisions to create.
        :param color_map: Optional colormap for the plot.
        :param border: Whether to plot a border around the simplex heat-map.
        :param ax: Optional matplotlib axes to plot on.
        :param kwargs: Additional arguments for tricontourf method.
        """
        corners = array([[0, 0], [1, 0], [0.5, 0.75 ** 0.5]])
        triangle = Triangulation(corners[:, 0], corners[:, 1])
        mid_points = [
            (corners[(i + 1) % 3] + corners[(i + 2) % 3]) / 2
            for i in range(3)
        ]

        def to_barycentric(cartesian):
            """
            Converts 2D Cartesian to barycentric coordinates.

            :param cartesian: A length-2 sequence containing the x and y value.
            """
            s = [(corners[i] - mid_points[i]).dot(
                     cartesian - mid_points[i]
                 ) / 0.75
                 for i in range(3)]
            s_clipped = clip(a=s, a_min=0, a_max=1)
            return s_clipped / norm(s_clipped, ord=1)

        refiner = UniformTriRefiner(triangle)
        tri_mesh = refiner.refine_triangulation(subdiv=num_sub_div)
        f = [self._method(to_barycentric(xy))
             for xy in zip(tri_mesh.x, tri_mesh.y)]
        ax = ax or new_axes()
        ax.tricontourf(tri_mesh, f, num_contours, cmap=color_map, **kwargs)
        ax.set_aspect('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.75 ** 0.5)
        ax.set_axis_off()
        if border:
            ax.triplot(triangle, linewidth=1)

        return ax
