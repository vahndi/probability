from typing import Union, List, Optional

from matplotlib.figure import Figure
from pandas import Series, DataFrame

from probability.custom_types.compound_types import RVMixin
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

    @staticmethod
    def infer_posterior(
            data: Series
    ) -> Union[RVContinuous1dMixin, RVDiscrete1dMixin]:
        """
        Return a new distribution of the posterior most likely to
        generate the given data.

        :param data: Series of observed values.
        """
        raise NotImplementedError

    @staticmethod
    def infer_posteriors(
            data: DataFrame,
            prob_vars: Union[str, List[str]],
            cond_vars: Union[str, List[str]],
            stats: Optional[Union[str, List[str]]] = None
    ) -> DataFrame:
        """
        Return a DataFrame mapping probability and conditional variables to
        Dirichlet distributions of posteriors most likely to generate the given
        data.

        :param data: DataFrame containing observed data.
        :param prob_vars: Name(s) of likelihood variables whose posteriors to
                          find probability of.
        :param cond_vars: Names of discrete variables to condition on.
                          Calculations will be done for the cartesian product
                          of variable values
                          e.g if cA = {1, 2} and cB = {3, 4} then
                          cAB = {(1,3), (1, 4), (2, 3), (2, 4)}.
        :param stats: Optional stats to append to the output e.g. 'alpha',
                      'mean'.
        :return: DataFrame with columns for each conditioning variable, a
                 'prob_var' column indicating the probability variable, and a
                 column named after the likelihood distribution containing the
                 distribution.
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
