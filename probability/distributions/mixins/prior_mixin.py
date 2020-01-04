from matplotlib.axes import Axes
from numpy import ndarray
from typing import Optional

from probability.custom_types import RVMixin
from probability.plots import new_axes


class PriorMixin(object):

    _distribution: RVMixin
    _x_label: str
    _y_label: str
    _label: str
    _str_params: str

    def plot(self, x: ndarray, color: str = 'C1', ax: Optional[Axes] = None) -> Axes:
        """
        Plot the prior probability of the parameter θ given the priors α, β

        `p(x|α,β)`

        :param x: vector of possible `x`s
        :param color: Optional color for the series.
        :param ax: Optional matplotlib axes
        :rtype: Axes
        """
        ax = ax or new_axes()
        self._distribution.pdf().at(x).plot(
            kind='line', label=self._label,
            color=color or 'C0', ax=ax
        )
        ax.set_xlabel(self._x_label)
        ax.set_ylabel(self._y_label)
        ax.legend()
        return ax
