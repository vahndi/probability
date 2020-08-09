from typing import Union

from matplotlib.figure import Figure

from probability.custom_types import RVMixin
from probability.distributions.mixins.rv_continuous_1d_mixin import \
    RVContinuous1dMixin
from probability.distributions.mixins.rv_discrete_1d_mixin import \
    RVDiscrete1dMixin


class ConjugateMixin(object):

    def prior(self, **kwargs) -> RVMixin:
        """
        Return a distribution reflecting the prior belief about the distribution
        of the parameters, before seeing any data.
        """
        raise NotImplementedError

    def likelihood(self, **kwargs) -> RVMixin:
        """
        Return a distribution reflecting the likelihood of observing the data,
        under the given type of model, independent of the prior belief about the
        distribution of parameters.
        """
        raise NotImplementedError

    def posterior(self, **kwargs) -> RVMixin:
        """
        Return a distribution reflecting the posterior belief about the
        distribution of the parameters, after observing the data.
        """
        raise NotImplementedError

    def plot(self, **kwargs) -> Figure:
        """
        Return a figure with the prior, likelihood, posterior and predictive
        distributions, if they exist.
        """
        raise NotImplementedError


class PredictiveMixin(object):
    """
    Used for conjugates that have a prior or posterior predictive distribution.
    """
    def prior_predictive(
            self, **kwargs
    ) -> Union[RVContinuous1dMixin, RVDiscrete1dMixin]:
        """
        Return a distribution that can be used to predict new data given the
        prior beliefs about the probability of the parameters before observing
        any data.
        """
        raise NotImplementedError

    def posterior_predictive(
            self, **kwargs
    ) -> Union[RVContinuous1dMixin, RVDiscrete1dMixin]:
        """
        Return a distribution that can be used to predict new data given the
        prior beliefs about the probability of the parameters after observing
        the data.
        """
        raise NotImplementedError


class AlphaFloatMixin(object):

    _alpha: float

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        self._alpha = value


class BetaFloatMixin(object):

    _beta: float

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, value: float):
        self._beta = value


class NIntMixin(object):

    _n: int

    @property
    def n(self) -> float:
        return self._n

    @n.setter
    def n(self, value: int):
        self._n = value


class KIntMixin(object):

    _k: int

    @property
    def k(self) -> float:
        return self._k

    @k.setter
    def k(self, value: int):
        self._k = value
