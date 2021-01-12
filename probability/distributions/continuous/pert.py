from typing import Optional

from scipy.stats import beta as beta_dist, rv_continuous

from compound_types.built_ins import FloatIterable
from probability.distributions.mixins.attributes import AFloatDMixin, \
    BFloatDMixin, CFloatDMixin
from probability.distributions.mixins.calculable_mixins import CalculableMixin
from probability.distributions.mixins.rv_continuous_1d_mixin import \
    RVContinuous1dMixin
from probability.utils import num_format


class PERT(
    RVContinuous1dMixin,
    AFloatDMixin,
    BFloatDMixin,
    CFloatDMixin,
    CalculableMixin,
    object
):
    """
    In probability and statistics, the PERT distribution is a family of
    continuous probability distributions defined by the minimum (a),
    most likely (b) and maximum (c) values that a variable can take.
    It is a transformation of the four-parameter Beta distribution with an
    additional assumption that its expected value is μ = (a + 4b + c) / 6

    The mean of the distribution is therefore defined as the weighted average of
    the minimum, most likely and maximum values that the variable may take,
    with four times the weight applied to the most likely value. This assumption
    about the mean was first proposed in Clark for estimating the effect of
    uncertainty of task durations on the outcome of a project schedule being
    evaluated using the program evaluation and review technique, hence its name.
    The mathematics of the distribution resulted from the authors' desire to
    make the standard deviation equal to about 1/6th of the range.

    The PERT distribution is widely used in risk analysis to represent the
    uncertainty of the value of some quantity where one is relying on subjective
    estimates, because the three parameters defining the distribution are
    intuitive to the estimator. The PERT distribution is featured in most
    simulation software tools.

    https://en.wikipedia.org/wiki/PERT_distribution
    """

    def __init__(self, a: float, b: float, c: float):
        """
        Create a new beta distribution.

        :param a: The minimum value of the distribution.
        :param b: The most likely value of the distribution.
        :param c: The maximum value of the distribution.
        """
        self._a: float = a
        self._b: float = b
        self._c: float = c
        self._reset_distribution()

    @property
    def alpha(self) -> float:
        return 1 + 4 * (self._b - self._a) / (self._c - self._a)

    @property
    def beta(self) -> float:
        return 1 + 4 * (self._c - self._b) / (self._c - self._a)

    def _reset_distribution(self):
        self._distribution: rv_continuous = beta_dist(
            a=self.alpha, b=self.beta,
            loc=self._a, scale=self._c - self._a
        )

    def mode(self) -> float:

        return self._b

    @property
    def lower_bound(self) -> float:
        return self._a

    @property
    def upper_bound(self) -> float:
        return self._c

    @staticmethod
    def fit(data: FloatIterable,
            a: Optional[float] = None,
            b: Optional[float] = None,
            c: Optional[float] = None) -> 'PERT':
        """
        Fit a PERT distribution to the data.

        :param data: Iterable of data to fit to.
        :param a: Optional fixed value for a.
        :param b: Optional fixed value for b.
        :param c: Optional fixed value for c.
        """
        kwargs = {}
        if a is not None:
            kwargs['floc'] = a
        if a is not None and c is not None:
            kwargs['fscale'] = c - a
        alpha, beta, loc, scale = beta_dist.fit(data=data, **kwargs)
        a = a if a is not None else loc
        c = c if c is not None else loc + scale
        if b is None:
            b_est_1 = a + (alpha * (c - a) - 1) / 4
            b_est_2 = c - (beta * (c - a) - 1) / 4
            b = (b_est_1 + b_est_2) / 2
        return PERT(a=a, b=b, c=c)

    def __str__(self):

        return f'PERT(' \
               f'a={num_format(self._a, 3)}, ' \
               f'b={num_format(self._b, 3)}, ' \
               f'c={num_format(self._c, 3)})'

    def __repr__(self):

        return f'PERT(a={self._a}, b={self._b}, c={self._c})'

    def __eq__(self, other: 'PERT') -> bool:

        return (
            abs(self._a - other._a) < 1e-10 and
            abs(self._b - other._b) < 1e-10 and
            abs(self._c - other._c) < 1e-10
        )

    def __ne__(self, other: 'PERT') -> bool:

        return not self.__eq__(other)
