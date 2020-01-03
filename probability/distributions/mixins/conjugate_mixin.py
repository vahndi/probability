from numpy import ndarray

from probability.custom_types import RVMixin


class ConjugateMixin(object):

    _prior: RVMixin
    _likelihood: RVMixin
    _posterior: RVMixin

    def _calculate_prior(self):
        raise NotImplementedError

    def _calculate_likelihood(self):
        raise NotImplementedError

    def _calculate_posterior(self):
        raise NotImplementedError

    # region component distributions

    @property
    def prior(self) -> RVMixin:
        raise NotImplementedError

    @property
    def likelihood(self) -> RVMixin:
        raise NotImplementedError

    @property
    def posterior(self) -> RVMixin:
        raise NotImplementedError

    # endregion

    # region plots

    def plot_prior(self, support=None):
        raise NotImplementedError

    def plot_likelihood(self, support=None):
        raise NotImplementedError

    def plot_posterior(self, support=None):
        raise NotImplementedError

    # end region

    # region calcs

    def posterior_mean(self):
        raise NotImplementedError

    def posterior_hpd(self):
        raise NotImplementedError

    # end region
