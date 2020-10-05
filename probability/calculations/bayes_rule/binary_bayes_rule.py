from typing import Union

from pandas import Series

from probability.calculations.bayes_rule.bayes_rule import BayesRule
from probability.calculations.context import sync_context
from probability.calculations.mixins import ProbabilityCalculationMixin
from probability.custom_types.external_custom_types import AnyFloatMap, \
    IntFloatMap
from probability.custom_types.internal_custom_types import AnyBetaMap
from probability.custom_types.type_checking import is_any_float_map, \
    is_any_beta_map
from probability.distributions import Beta


class BinaryBayesRule(BayesRule):
    """
    Class for testing binary hypotheses with Bayes Rule.
    """
    def __init__(
            self,
            prior: Union[float, Beta, AnyFloatMap],
            likelihood: Union[float, Beta, AnyFloatMap, AnyBetaMap]
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
        self._prior: Union[float, Beta, AnyFloatMap] = prior
        self._likelihood: Union[
            float, Beta, AnyFloatMap, AnyBetaMap
        ] = likelihood

    @staticmethod
    def _posterior__p_f__l_f(
            prior: float, likelihood: float
    ) -> float:
        """
        Calculate the single-figure posterior probability.

        :param prior: Single-figure prior probability.
        :param likelihood: Single-figure likelihood.
        :return: Single-figure posterior probability.
        """
        lp_1 = prior * likelihood
        lp_0 = (1 - prior) * (1 - likelihood)
        return lp_1 / (lp_1 + lp_0)

    @staticmethod
    def _posterior__p_f__l_fm(
            prior: float, likelihood: AnyFloatMap
    ) -> Union[Series, AnyFloatMap]:
        """
        Calculate a Series of single figure posterior
        probabilities for each likelihood value.

        :param prior: Single-figure prior probability.
        :param likelihood: Series of single-figure likelihood values, with
                           likelihood categories as index.
        :return: Series of single-figure posterior probabilities.
        """
        return Series({
            key: BinaryBayesRule._posterior__p_f__l_f(prior, value)
            for key, value in likelihood.items()
        })

    @staticmethod
    def _posterior__p_f__l_b(
            prior: float, likelihood: Beta,
    ) -> Union[Series, IntFloatMap]:
        """
        Calculate a Series of sampled posterior probabilities from a
        single-figure prior and Beta-distributed likelihood.

        :param prior: Single figure prior probability.
        :param likelihood: Beta-distributed likelihood.
        :return: Series of posterior probability calculations.
        """
        lp_1 = prior * likelihood
        lp_0 = (1 - prior) * (1 - likelihood)
        posterior = lp_1 / (lp_1 + lp_0)
        sync_context(posterior)
        return posterior

    @staticmethod
    def _posterior__p_f__l_bm(
            prior: float, likelihood: AnyBetaMap,
    ) -> Series:
        """
        Calculate a DataFrame of sampled posterior probabilities from a single-
        figure prior and a series of Beta-distributed likelihoods.

        :param prior: Single-figure prior probability.
        :param likelihood: Series of Beta-distributed likelihoods.
        :return: Series of posterior probability calculations.
        """
        lp_1 = prior * likelihood
        lp_0 = (1 - prior) * (1 - likelihood)
        posterior = lp_1 / (lp_1 + lp_0)
        sync_context(posterior)
        return posterior

    @staticmethod
    def _posterior__p_b__l_f(
            prior: Beta, likelihood: float,
    ) -> ProbabilityCalculationMixin:
        """
        Calculate a Series of sampled posterior probabilities from a
        Beta-distributed prior and a single figure likelihood.

        :param prior: Beta distributed prior probability.
        :param likelihood: Single-figure likelihood.
        :return: Posterior probability calculation.
        """
        lp_1: ProbabilityCalculationMixin = prior * likelihood
        lp_0: ProbabilityCalculationMixin = (1 - prior) * (1 - likelihood)
        posterior = lp_1 / (lp_1 + lp_0)
        sync_context(posterior)
        return posterior

    @staticmethod
    def _posterior__p_b__l_fm(
            prior: Beta, likelihood: AnyFloatMap,
    ) -> Series:
        """
        Calculate a DataFrame of sampled posterior probabilities from a
        Beta-distributed prior and a Series of single figure likelihoods.

        :param prior: Beta-distributed prior probability.
        :param likelihood: Series of single-figure likelihoods.
        :return: Series of posterior probability calculations.
        """
        lp_1: ProbabilityCalculationMixin = prior * likelihood
        lp_0: ProbabilityCalculationMixin = (1 - prior) * (1 - likelihood)
        posterior = lp_1 / (lp_1 + lp_0)
        sync_context(posterior)
        return posterior

    @staticmethod
    def _posterior__p_b__l_b(
            prior: Beta, likelihood: Beta,
    ) -> ProbabilityCalculationMixin:
        """
        Return a Series of posterior probabilities sampled from a
        Beta-distributed prior and likelihood.

        :param prior: Beta-distributed prior probability.
        :param likelihood: Beta-distributed likelihood.
        :return: Posterior probability calculation.
        """
        lp_1: ProbabilityCalculationMixin = prior * likelihood
        lp_0: ProbabilityCalculationMixin = (1 - prior) * (1 - likelihood)
        posterior = lp_1 / (lp_1 + lp_0)
        sync_context(posterior)
        return posterior

    @staticmethod
    def _posterior__p_b__l_bm(
            prior: Beta, likelihood: AnyBetaMap,
    ) -> Series:
        """
        Calculate a DataFrame of sampled posterior probabilities from a
        Beta-distributed prior and a Series of Beta-distributed likelihoods.

        :param prior: Beta-distributed prior probability.
        :param likelihood: Series of Beta-distributed likelihoods.
        :return: Series of posterior probability calculations.
        """
        lp_1: ProbabilityCalculationMixin = prior * likelihood
        lp_0: ProbabilityCalculationMixin = (1 - prior) * (1 - likelihood)
        posterior = lp_1 / (lp_1 + lp_0)
        sync_context(posterior)
        return posterior

    @staticmethod
    def _posterior__p_fm__l__fm(
            prior: AnyFloatMap, likelihood: AnyFloatMap,
    ) -> Union[AnyFloatMap, Series]:
        """
        Calculate a Series of single-figure posterior probabilities from a
        Series of float priors and Series of float likelihoods.

        :param prior: Series of float priors.
        :param likelihood: Series of float likelihoods.
        :return: Series of single figure probabilities.
        """
        lp_1: Series = prior * likelihood
        lp_0: Series = (1 - prior) * (1 - likelihood)
        posterior = lp_1 / (lp_1 + lp_0)
        return posterior

    def posterior(self) -> Union[float, ProbabilityCalculationMixin, Series]:
        """
        Return samples from the posterior P(A|B).
        Columns are tuples of (a, b).
        """
        if isinstance(self._prior, float):
            if isinstance(self._likelihood, float):
                return BinaryBayesRule._posterior__p_f__l_f(
                    prior=self._prior,
                    likelihood=self._likelihood
                )
            elif is_any_float_map(self._likelihood):
                return BinaryBayesRule._posterior__p_f__l_fm(
                    prior=self._prior,
                    likelihood=self._likelihood
                )
            elif isinstance(self._likelihood, Beta):
                return BinaryBayesRule._posterior__p_f__l_b(
                    prior=self._prior,
                    likelihood=self._likelihood,
                )
            elif is_any_beta_map(self._likelihood):
                return BinaryBayesRule._posterior__p_f__l_bm(
                    prior=self._prior,
                    likelihood=self._likelihood,
                )
        elif isinstance(self._prior, Beta):
            if isinstance(self._likelihood, float):
                return BinaryBayesRule._posterior__p_b__l_f(
                    prior=self._prior,
                    likelihood=self._likelihood,
                )
            elif is_any_float_map(self._likelihood):
                return BinaryBayesRule._posterior__p_b__l_fm(
                    prior=self._prior,
                    likelihood=self._likelihood,
                )
            elif isinstance(self._likelihood, Beta):
                return BinaryBayesRule._posterior__p_b__l_b(
                    prior=self._prior,
                    likelihood=self._likelihood,
                )
            elif is_any_beta_map(self._likelihood):
                return BinaryBayesRule._posterior__p_b__l_bm(
                    prior=self._prior,
                    likelihood=self._likelihood,
                )
        elif is_any_float_map(self._prior):
            if is_any_float_map(self._likelihood):
                return BinaryBayesRule._posterior__p_fm__l__fm(
                    prior=self._prior,
                    likelihood=self._likelihood,
                )
        else:
            raise TypeError(
                'wrong combination of types for prior and likelihood'
            )
