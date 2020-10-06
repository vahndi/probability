from typing import Union

from pandas import Series

from probability.calculations.bayes_rule.bayes_rule import BayesRule
from probability.calculations.context import sync_context
from probability.calculations.mixins import ProbabilityCalculationMixin
from probability.custom_types.external_custom_types import AnyFloatMap
from probability.custom_types.internal_custom_types import AnyBetaMap, \
    AnyCalculationMap
from probability.custom_types.type_checking import is_any_beta_map, \
    is_any_numeric_map
from probability.distributions import Beta


class BinaryBayesRule(BayesRule):
    """
    Class for testing one or more binary hypotheses using Bayes Rule.
    """
    def __init__(
            self,
            prior: Union[float, Beta, AnyFloatMap, AnyBetaMap],
            likelihood: Union[float, Beta, AnyFloatMap, AnyBetaMap]
    ):
        """
        Create a new Bayes Rule object from:
          - the prior P(A)
          - the likelihood P(B|A)
          - the evidence P(B)

        :param prior: Single figure or Beta-distributed probability representing
                      the hypothesis.
        :param likelihood: Series with values of Dirichlet likelihoods.
        """
        if not (
            isinstance(prior, float) or
            isinstance(prior, int) or
            isinstance(prior, Beta) or
            is_any_numeric_map(prior) or
            is_any_beta_map(prior)
        ):
            raise ValueError('wrong type for prior')
        if not (
            isinstance(likelihood, float) or
            isinstance(likelihood, int) or
            isinstance(likelihood, Beta) or
            is_any_numeric_map(likelihood) or
            is_any_beta_map(likelihood)
        ):
            raise ValueError('wrong type for prior')

        self._prior: Union[
            float, Beta, AnyFloatMap, AnyBetaMap
        ] = prior
        self._likelihood: Union[
            float, Beta, AnyFloatMap, AnyBetaMap
        ] = likelihood

    def posterior(self) -> Union[
        float, ProbabilityCalculationMixin, Series, AnyCalculationMap
    ]:
        """
        Return posterior probability P(A|B).
        """
        lp_1 = self._prior * self._likelihood
        lp_0 = (1 - self._prior) * (1 - self._likelihood)
        posterior = lp_1 / (lp_1 + lp_0)
        if not isinstance(posterior, float):
            sync_context(posterior)
        return posterior
