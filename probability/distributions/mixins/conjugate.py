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


