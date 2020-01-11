from numpy import ndarray
from typing import overload, Union, Iterable

from probability.custom_types import RVMixin


class ConjugateMixin(object):

    # region component distributions

    def prior(self, **kwargs) -> RVMixin:
        """
        Return a distribution reflecting the prior belief about the distribution of the parameters,
        before seeing any data.
        """
        raise NotImplementedError

    def likelihood(self, **kwargs) -> RVMixin:
        """
        Return a distribution reflecting the likelihood of observing the data, under the given type of model,
        independent of the prior belief about the distribution of parameters.
        """
        raise NotImplementedError

    def posterior(self, **kwargs) -> RVMixin:
        """
        Return a distribution reflecting the posterior belief about the distribution of the parameters,
        after observing the data.
        """
        raise NotImplementedError

    # endregion

    @overload
    def predict_proba(self, m: float) -> float:
        pass

    @overload
    def predict_proba(self, n: Union[ndarray, Iterable]) -> ndarray:
        pass

    def predict_proba(self, **kwargs):
        """
        Predict the probability of observing one or more data points given the posterior distribution.
        
        e.g. for the beta-binomial model this is the pmf of the beta-binomial distribution
        (n choose k) B(k + a, n-k + b) / B(a, b) where B is the beta function
        """
        raise NotImplementedError
