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
        self._prior: Union[float, Beta, AnyFloatMap] = prior
        self._likelihood: Union[
            float, Beta, AnyFloatMap, AnyBetaMap
        ] = likelihood

    @staticmethod
    def _posterior__p_f__l_f(
            prior: float, likelihood: float
    ) -> float:
        """
        Calculate a single-figure posterior probability.

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
    ) -> ProbabilityCalculationMixin:
        """
        Calculate a posterior probability calculation from a
        single-figure prior and Beta-distributed likelihood.

        :param prior: Single figure prior probability.
        :param likelihood: Beta-distributed likelihood.
        :return: Posterior probability calculation.
        """
        lp_1 = prior * likelihood
        lp_0 = (1 - prior) * (1 - likelihood)
        posterior = lp_1 / (lp_1 + lp_0)
        sync_context(posterior)
        return posterior

    @staticmethod
    def _posterior__p_f__l_bm(
            prior: float, likelihood: AnyBetaMap,
    ) -> Union[Series, AnyCalculationMap]:
        """
        Calculate a Series of sampled posterior probabilities from a single-
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
    ) -> Union[Series, AnyCalculationMap]:
        """
        Calculate a Series of posterior calculations from a
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
        Calculate a Series of posterior probabilities sampled from a
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
    ) -> Union[Series, AnyCalculationMap]:
        """
        Calculate a Series of sampled posterior probabilities from a
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
    def _posterior__p_fm__l_f(
            prior: AnyFloatMap, likelihood: float
    ) -> Union[Series, AnyFloatMap]:
        """
        Calculate a Series of single figure posterior probabilities from a
        Series of single figure priors and a single figure likelihood.

        :param prior: Series of single figure prior probabilities.
        :param likelihood: Single figure likelihood.
        :return: Series of single-figure posterior probabilities.
        """
        lp_1: Series = prior * likelihood
        lp_0: Series = (1 - prior) * (1 - likelihood)
        posterior = lp_1 / (lp_1 + lp_0)
        return posterior

    @staticmethod
    def _posterior__p_fm__l_fm(
            prior: AnyFloatMap, likelihood: AnyFloatMap,
    ) -> Union[AnyFloatMap, Series]:
        """
        Calculate a Series of single-figure posterior probabilities from a
        Series of float priors and Series of float likelihoods.

        :param prior: Series of float priors.
        :param likelihood: Series of float likelihoods.
        :return: Series of single figure posterior probabilities.
        """
        lp_1: Series = prior * likelihood
        lp_0: Series = (1 - prior) * (1 - likelihood)
        posterior = lp_1 / (lp_1 + lp_0)
        return posterior

    @staticmethod
    def _posterior__p_fm__l_b(
            prior: AnyFloatMap, likelihood: Beta
    ) -> AnyCalculationMap:
        """
        Calculate a Series of posterior probability calculations from a
        Series of float priors and a Beta-distributed likelihood.

        :param prior: Series of float priors.
        :param likelihood: Series of float likelihoods.
        :return: Series of single figure posterior probabilities.
        """
        lp_1: Series = prior * likelihood
        lp_0: Series = (1 - prior) * (1 - likelihood)
        posterior = lp_1 / (lp_1 + lp_0)
        sync_context(posterior)
        return posterior

    @staticmethod
    def _posterior__p_fm__l_bm(
            prior: AnyFloatMap, likelihood: AnyBetaMap
    ) -> AnyCalculationMap:
        """
        Calculate a Series of posterior probability calculations from a
        Series of float priors and a Series of Beta-distributed likelihoods.

        :param prior: Series of float priors.
        :param likelihood: Series of Beta-distributed likelihoods.
        :return: Series of posterior probability calculations.
        """
        lp_1: Series = prior * likelihood
        lp_0: Series = (1 - prior) * (1 - likelihood)
        posterior = lp_1 / (lp_1 + lp_0)
        sync_context(posterior)
        return posterior

    @staticmethod
    def _posterior__p_bm__l_f(
            prior: AnyBetaMap, likelihood: float
    ) -> AnyCalculationMap:
        """
        Calculate a Series of posterior probability calculations from a
        Series of Beta-distributed priors and a single figure likelihood.

        :param prior: Series of Beta-distributed priors.
        :param likelihood: Single figure likelihood.
        :return: Series of posterior probability calculations.
        """
        lp_1: Series = prior * likelihood
        lp_0: Series = (1 - prior) * (1 - likelihood)
        posterior = lp_1 / (lp_1 + lp_0)
        sync_context(posterior)
        return posterior

    @staticmethod
    def _posterior__p_bm__l_fm(
            prior: AnyBetaMap, likelihood: AnyFloatMap
    ) -> AnyCalculationMap:
        """
        Calculate a Series of posterior probability calculations from a
        Series of Beta-distributed priors and a single figure likelihood.

        :param prior: Series of Beta-distributed priors.
        :param likelihood: Series of single figure likelihoods.
        :return: Series of posterior probability calculations.
        """
        lp_1: Series = prior * likelihood
        lp_0: Series = (1 - prior) * (1 - likelihood)
        posterior = lp_1 / (lp_1 + lp_0)
        sync_context(posterior)
        return posterior

    @staticmethod
    def _posterior__p_bm__l_b(
            prior: AnyBetaMap, likelihood: Beta
    ) -> AnyCalculationMap:
        """
        Calculate a Series of posterior probability calculations from a
        Series of Beta-distributed priors and a single figure likelihood.

        :param prior: Series of Beta-distributed priors.
        :param likelihood: Beta-distributed likelihood.
        :return: Series of posterior probability calculations.
        """
        lp_1: Series = prior * likelihood
        lp_0: Series = (1 - prior) * (1 - likelihood)
        posterior = lp_1 / (lp_1 + lp_0)
        sync_context(posterior)
        return posterior

    @staticmethod
    def _posterior__p_bm__l_bm(
            prior: AnyBetaMap, likelihood: AnyBetaMap
    ) -> AnyCalculationMap:
        """
        Calculate a Series of posterior probability calculations from a
        Series of Beta-distributed priors and a single figure likelihood.

        :param prior: Series of Beta-distributed priors.
        :param likelihood: Series of Beta-distributed likelihoods.
        :return: Series of posterior probability calculations.
        """
        lp_1: Series = prior * likelihood
        lp_0: Series = (1 - prior) * (1 - likelihood)
        posterior = lp_1 / (lp_1 + lp_0)
        sync_context(posterior)
        return posterior

    def posterior(self) -> Union[
        float, ProbabilityCalculationMixin, Series, AnyCalculationMap
    ]:
        """
        Return samples from the posterior P(A|B).
        Columns are tuples of (a, b).
        """
        if (
                isinstance(self._prior, float) or
                isinstance(self._prior, int)
        ):
            if (
                    isinstance(self._likelihood, float) or
                    isinstance(self._likelihood, int)
            ):
                return BinaryBayesRule._posterior__p_f__l_f(
                    prior=self._prior,
                    likelihood=self._likelihood
                )
            elif is_any_numeric_map(self._likelihood):
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
            if (
                    isinstance(self._likelihood, float) or
                    isinstance(self._likelihood, int)
            ):
                return BinaryBayesRule._posterior__p_b__l_f(
                    prior=self._prior,
                    likelihood=self._likelihood,
                )
            elif is_any_numeric_map(self._likelihood):
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
        elif is_any_numeric_map(self._prior):
            if (
                    isinstance(self._likelihood, float) or
                    isinstance(self._likelihood, int)
            ):
                return BinaryBayesRule._posterior__p_fm__l_f(
                    prior=self._prior,
                    likelihood=self._likelihood
                )
            elif is_any_numeric_map(self._likelihood):
                return BinaryBayesRule._posterior__p_fm__l_fm(
                    prior=self._prior,
                    likelihood=self._likelihood
                )
            elif isinstance(self._likelihood, Beta):
                return BinaryBayesRule._posterior__p_fm__l_b(
                    prior=self._prior,
                    likelihood=self._likelihood
                )
            elif is_any_beta_map(self._likelihood):
                return BinaryBayesRule._posterior__p_fm__l_bm(
                    prior=self._prior,
                    likelihood=self._likelihood
                )
        elif is_any_beta_map(self._prior):
            if (
                    isinstance(self._likelihood, float) or
                    isinstance(self._likelihood, int)
            ):
                return BinaryBayesRule._posterior__p_bm__l_f(
                    prior=self._prior,
                    likelihood=self._likelihood
                )
            elif is_any_numeric_map(self._likelihood):
                return BinaryBayesRule._posterior__p_bm__l_fm(
                    prior=self._prior,
                    likelihood=self._likelihood
                )
            elif isinstance(self._likelihood, Beta):
                return BinaryBayesRule._posterior__p_bm__l_b(
                    prior=self._prior,
                    likelihood=self._likelihood
                )
            elif is_any_beta_map(self._likelihood):
                return BinaryBayesRule._posterior__p_bm__l_bm(
                    prior=self._prior,
                    likelihood=self._likelihood
                )
        else:
            raise TypeError(
                'wrong combination of types for prior and likelihood'
            )
