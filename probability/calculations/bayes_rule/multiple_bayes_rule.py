from typing import Union

from pandas import Series, DataFrame

from probability.calculations.bayes_rule.bayes_rule import BayesRule
from probability.calculations.context import sync_context
from probability.calculations.mixins import ProbabilityCalculationMixin
from probability.custom_types.external_custom_types import AnyFloatMap
from probability.custom_types.internal_custom_types import AnyDirichletMap
from probability.custom_types.type_checking import is_any_dirichlet_map, \
    is_any_numeric_map
from probability.distributions import Dirichlet


class MultipleBayesRule(BayesRule):
    """
    Class for testing multiple hypotheses with Bayes Rule.
    """
    def __init__(
            self,
            prior: Union[AnyFloatMap, Dirichlet],
            likelihood: Union[Dirichlet, AnyFloatMap, AnyDirichletMap]
    ):
        """
        Create a new Bayes Rule object from:
          - the prior P(A)
          - the likelihood P(B|A)
          - the evidence P(B)

        :param prior: Dirichlet where each dimension represents one likelihood
                      category.
        :param likelihood: Series with values of Dirichlet likelihoods.
        """
        self._prior: Union[AnyFloatMap, Dirichlet] = prior
        self._likelihood: Union[
            Dirichlet, AnyFloatMap, AnyDirichletMap
        ] = likelihood

    @staticmethod
    def from_counts(
            data: DataFrame,
            prior_weight: float = 1.0
    ) -> 'MultipleBayesRule':
        """
        Return a new BayesRule class using a DataFrame of counts.

        :param data: DataFrame of counts where the index represents the
                     different states of the evidence B, and each column
                     represents the likelihood for one value of the prior A.
        :param prior_weight: Proportion of the overall counts to use as the
                             prior probability.
        """
        prior = Dirichlet(1 + data.sum() * prior_weight)
        likelihood = Series({
            key: Dirichlet(1 + row)
            for key, row in data.iterrows()
        })
        return MultipleBayesRule(prior=prior, likelihood=likelihood)

    @staticmethod
    def _posterior__p_fm__l__fm(
            prior: AnyFloatMap, likelihood: AnyFloatMap,
    ) -> Union[AnyFloatMap, Series]:
        """
        Calculate a Series of single-figure posterior probabilities from a
        Series of float priors and Series of float likelihoods.

        N.B. assumes each key corresponds to a hypothesis. To calculate a
        sequence of posteriors for different experiments, create a sequence of
        BayesRule instances, one for each prior.

        :param prior: Series of float priors, one per hypothesis.
        :param likelihood: Series of float likelihoods, one per hypothesis.
        :return: Series of single figure posterior probabilities,
                 one per hypothesis.
        """
        if not set(prior.keys()) == set(likelihood.keys()):
            raise ValueError('keys of prior and float must be the same')
        if isinstance(prior, dict):
            prior = Series(prior)
        if isinstance(likelihood, dict):
            likelihood = Series(likelihood)
        if not prior.sum() == 1:
            raise ValueError('sum of priors must be 1')

        lp = prior * likelihood
        normalization = lp.sum()
        return lp / normalization

    @staticmethod
    def _posterior__p_d__l_d(
        prior: Dirichlet, likelihood: Dirichlet,
    ) -> ProbabilityCalculationMixin:
        """
        Return a DataFrame of posterior probabilities sampled from a
        Dirichlet-distributed prior and likelihood.

        :param prior: Dirichlet-distributed prior probability.
        :param likelihood: Dirichlet-distributed likelihood.
        :return: Posterior probability calculation.
        """
        numerator = prior * likelihood
        denominator = numerator.sum()
        posterior = numerator / denominator
        sync_context(posterior)
        return posterior

    @staticmethod
    def _posterior__p_d__l_dm(
            prior: Dirichlet, likelihood: AnyDirichletMap,
    ) -> Series:
        """
        Return a DataFrame of posterior probabilities sampled from a
        Dirichlet-distributed prior and a Series of Dirichlet-distributed
        likelihoods.

        :param prior: Dirichlet-distributed prior probability.
        :param likelihood: Dirichlet-distributed likelihood.
        :return: Series of posterior probability calculations.
        """
        numerators = prior * likelihood
        denominators = numerators.map(lambda c: c.sum())
        posteriors = numerators / denominators
        sync_context(posteriors)
        return posteriors

    def posterior(self) -> Union[float, ProbabilityCalculationMixin, Series]:
        """
        Return samples from the posterior P(A|B).
        Columns are tuples of (a, b).
        """
        if is_any_numeric_map(self._prior):
            if is_any_numeric_map(self._likelihood):
                return MultipleBayesRule._posterior__p_fm__l__fm(
                    prior=self._prior,
                    likelihood=self._likelihood,
                )
        elif isinstance(self._prior, Dirichlet):
            if isinstance(self._likelihood, Dirichlet):
                return MultipleBayesRule._posterior__p_d__l_d(
                    prior=self._prior,
                    likelihood=self._likelihood,
                )
            elif is_any_dirichlet_map(self._likelihood):
                return MultipleBayesRule._posterior__p_d__l_dm(
                    prior=self._prior,
                    likelihood=self._likelihood,
                )
        else:
            raise TypeError(
                'wrong combination of types for prior and likelihood'
            )
