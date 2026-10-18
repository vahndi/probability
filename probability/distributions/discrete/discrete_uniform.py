from typing import Union, Tuple

from scipy.stats import randint, rv_discrete

from probability.custom_types.external_custom_types import FloatIterable
from probability.distributions.mixins.rv_discrete_1d_mixin import (
    RVDiscrete1dMixin,
)
from probability.utils import num_format, is_scalar


class DiscreteUniform(
    RVDiscrete1dMixin,
    object
):
    """
    The discrete uniform distribution is a symmetric probability distribution
    wherein a finite number of integers are equally likely to be observed.
    Every one of n values has equal probability 1/n.

    Parameters:
    - a: Lower integer bound (inclusive).
    - b: Upper integer bound (inclusive, b >= a).

    https://en.wikipedia.org/wiki/Discrete_uniform_distribution
    """

    def __init__(self, a: int, b: int):
        """
        Create a new DiscreteUniform distribution on integers [a, b].

        :param a: Lower bound (inclusive).
        :param b: Upper bound (inclusive, b >= a).
        """
        if a > b:
            raise ValueError(f"a must be <= b, got a={a}, b={b}")
        self._a: int = int(a)
        self._b: int = int(b)
        self._reset_distribution()

    def _reset_distribution(self):
        self._distribution: rv_discrete = randint(
            low=self._a,
            high=self._b + 1
        )

    @property
    def a(self) -> int:
        return self._a

    @a.setter
    def a(self, value: int):
        if value > self._b:
            raise ValueError(f"a must be <= b, got a={value}, b={self._b}")
        self._a = int(value)
        self._reset_distribution()

    @property
    def b(self) -> int:
        return self._b

    @b.setter
    def b(self, value: int):
        if value < self._a:
            raise ValueError(f"b must be >= a, got b={value}, a={self._a}")
        self._b = int(value)
        self._reset_distribution()

    @property
    def n(self) -> int:
        """Number of possible outcomes: b - a + 1."""
        return self._b - self._a + 1

    @property
    def lower_bound(self) -> int:
        return self._a

    @property
    def upper_bound(self) -> int:
        return self._b

    def mode(self) -> Tuple[int, int]:
        return self._a, self._b

    @staticmethod
    def fit(data: FloatIterable) -> 'DiscreteUniform':
        """
        Fit a DiscreteUniform distribution to discrete observation data.

        :param data: Iterable of integer values.
        """
        data_list = [int(x) for x in data]
        if not data_list:
            raise ValueError("Data cannot be empty.")
        return DiscreteUniform(a=min(data_list), b=max(data_list))

    def __str__(self):
        return f'DiscreteUniform(a={self._a}, b={self._b})'

    def __repr__(self):
        return f'DiscreteUniform(a={self._a}, b={self._b})'

    def __eq__(self, other: Union['DiscreteUniform', int, float]) -> Union[bool, float]:
        if is_scalar(other):
            return self.pmf().at(other)
        elif isinstance(other, DiscreteUniform):
            return self._a == other._a and self._b == other._b
        return False

    def __ne__(self, other: Union['DiscreteUniform', int, float]) -> Union[bool, float]:
        if is_scalar(other):
            return 1.0 - float(self.__eq__(other))
        return not self.__eq__(other)
