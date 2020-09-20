from typing import Union, Optional

from pandas import DataFrame, Series

from probability.custom_types.external_custom_types import AnyFloatMap, \
    IntFloatMap
from probability.custom_types.internal_custom_types import AnyBetaMap, \
    AnyDirichletMap
from probability.custom_types.type_checking import is_any_dirichlet_map, \
    is_any_float_map, is_any_beta_map
from probability.distributions import Dirichlet, Beta
from probability.distributions.mixins.rv_mixins import NUM_SAMPLES_COMPARISON


class BayesRule(object):

    def __init__(
            self,
            prior: Union[float, Beta, AnyFloatMap, Dirichlet],
            likelihood: Union[float, AnyFloatMap, AnyBetaMap, AnyDirichletMap]
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
        self._prior: Union[float, Beta, AnyFloatMap, Dirichlet] = prior
        self._likelihood: Union[
            float, AnyFloatMap, AnyBetaMap, AnyDirichletMap
        ] = likelihood

    @staticmethod
    def from_counts(
            data: DataFrame,
            prior_weight: float = 1.0
    ) -> 'BayesRule':
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
        return BayesRule(prior=prior, likelihood=likelihood)

    @property
    def prior(self) -> Union[float, Beta, AnyFloatMap, Dirichlet]:

        return self._prior

    @property
    def likelihood(self) -> Union[
            float, AnyFloatMap, AnyBetaMap, AnyDirichletMap
    ]:

        return self._likelihood

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
        :return: Series of single-figure posterior probabilities with likelihood
                 categories as index.
        """
        return Series({
            key: BayesRule._posterior__p_f__l_f(prior, value)
            for key, value in likelihood.items()
        })

    @staticmethod
    def _posterior__p_f__l_b(
            prior: float, likelihood: Beta,
            num_samples: Optional[int] = None
    ) -> Union[Series, IntFloatMap]:
        """
        Calculate a Series of sampled posterior probabilities from a
        single-figure prior and Beta-distributed likelihood.

        :param prior: Single figure prior probability.
        :param likelihood: Beta-distributed likelihood.
        :return: Series of posterior probability samples with sample number
                 index.
        """
        if num_samples is None:
            num_samples = NUM_SAMPLES_COMPARISON
        p_like = likelihood.rvs(num_samples)
        lp_1: Series = p_like * prior
        lp_0: Series = (1 - p_like) * (1 - prior)
        return lp_1 / (lp_1 + lp_0)

    @staticmethod
    def _posterior__p_f__l_bm(
            prior: float, likelihood: AnyBetaMap,
            num_samples: Optional[int] = None
    ) -> DataFrame:
        """
        Calculate a DataFrame of sampled posterior probabilities from a single-
        figure prior and a series of Beta-distributed likelihoods.

        :param prior: Single-figure prior probability.
        :param likelihood: Series of Beta-distributed likelihoods.
        :param num_samples: Number of samples to draw from each likelihood.
        :return: DataFrame of posterior probabilities with sample number as
                 index and likelihood categories as columns.
        """
        posterior_samples = {}
        for like_name, like in likelihood.items():
            posterior_samples[like_name] = BayesRule._posterior__p_f__l_b(
                prior=prior, likelihood=likelihood,
                num_samples=num_samples
            )
        return DataFrame(posterior_samples)

    @staticmethod
    def _posterior__p_b__l_f(
            prior: Beta, likelihood: float,
            num_samples: Optional[int] = None
    ) -> Union[Series, IntFloatMap]:
        """
        Calculate a Series of sampled posterior probabilities from a
        Beta-distributed prior and a single figure likelihood.

        :param prior: Beta distributed prior probability.
        :param likelihood: Single-figure likelihood.
        :param num_samples: Number of samples to draw from the prior.
        :return: Series of posterior probability samples with sample number
                 index.
        """
        if num_samples is None:
            num_samples = NUM_SAMPLES_COMPARISON

        p_prior = prior.rvs(num_samples)
        lp_1: Series = likelihood * p_prior
        lp_0: Series = (1 - likelihood) * (1 - p_prior)
        return lp_1 / (lp_1 + lp_0)

    @staticmethod
    def _posterior__p_b__l_fm(
            prior: Beta, likelihood: AnyFloatMap,
            num_samples: Optional[int] = None
    ) -> DataFrame:
        """
        Calculate a DataFrame of sampled posterior probabilities from a
        Beta-distributed prior and a Series of single figure likelihoods.

        :param prior: Beta-distributed prior probability.
        :param likelihood: Series of single-figure likelihoods.
        :param num_samples: Number of samples to draw from the prior
                            distribution.
        :return: DataFrame of samples with a column for each likelihood
                 category.
        """
        posterior_samples = {}
        for like_name, like in likelihood.items():
            posterior_samples[like_name] = BayesRule._posterior__p_b__l_f(
                prior=prior, likelihood=like,
                num_samples=num_samples
            )
        return DataFrame(posterior_samples)

    @staticmethod
    def _posterior__p_b__l_b(
            prior: Beta, likelihood: Beta,
            num_samples: Optional[int] = None
    ):
        """
        Return a Series of posterior probabilities sampled from a
        Beta-distributed prior and likelihood.

        :param prior: Beta-distributed prior probability.
        :param likelihood: Beta-distributed likelihood.
        :param num_samples: Number of samples to draw from each distribution.
        :return: Series of posterior probability samples with sample number
                 index.
        """
        if num_samples is None:
            num_samples = NUM_SAMPLES_COMPARISON

        p_prior = prior.rvs(num_samples)
        p_like = likelihood.rvs(num_samples)
        lp_1: Series = p_like * p_prior
        lp_0: Series = (1 - p_like) * (1 - p_prior)
        return lp_1 / (lp_1 + lp_0)

    @staticmethod
    def _posterior__p_b__l_bm(
            prior: Beta, likelihood: AnyBetaMap,
            num_samples: Optional[int] = None
    ):
        """
        Calculate a DataFrame of sampled posterior probabilities from a
        Beta-distributed prior and a Series of Beta-distributed likelihoods.

        :param prior: Beta-distributed prior probability.
        :param likelihood: Series of Beta-distributed likelihoods.
        :param num_samples: Number of samples to draw from the prior
                            distribution.
        :return: DataFrame of samples with a column for each likelihood
                 category.
        """
        posterior_samples = {}
        for like_name, like in likelihood.items():
            posterior_samples[like_name] = BayesRule._posterior__p_b__l_b(
                prior=prior, likelihood=like,
                num_samples=num_samples
            )
        return DataFrame(posterior_samples)

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
        prior_states = list(prior.keys())
        posteriors = {}
        likelihoods = {
            state: prior[state] * likelihood[state]
            for state in prior_states
        }
        evidence = sum(likelihoods.values())
        for state in prior_states:
            posteriors[state] = likelihoods[state] / evidence
        return Series(posteriors)

    @staticmethod
    def _posterior__p_d__l_d(
        prior: Dirichlet, likelihood: Dirichlet,
        num_samples: Optional[int] = None
    ) -> DataFrame:
        """
        Return a DataFrame of posterior probabilities sampled from a
        Dirichlet-distributed prior and likelihood.

        :param prior: Dirichlet-distributed prior probability.
        :param likelihood: Dirichlet-distributed likelihood.
        :param num_samples: Number of samples to draw from each distribution.
        :return: DataFrame of posterior probability samples with sample number
                 index and prior categories as columns.
        """
        if num_samples is None:
            num_samples = NUM_SAMPLES_COMPARISON

        p_prior = prior.rvs(num_samples)
        p_likelihood = likelihood.rvs(num_samples)
        p_evidence = (p_prior * p_likelihood).sum(axis=1)
        posterior = (p_prior * p_likelihood).div(p_evidence, axis=0)
        return posterior

    @staticmethod
    def _posterior__p_d__l_dm(
            prior: Dirichlet, likelihood: AnyDirichletMap,
            num_samples: Optional[int] = None
    ) -> DataFrame:
        """
        Return a DataFrame of posterior probabilities sampled from a
        Dirichlet-distributed prior and a Series of Dirichlet-distributed
        likelihoods.

        :param prior: Dirichlet-distributed prior probability.
        :param likelihood: Dirichlet-distributed likelihood.
        :param num_samples: Number of samples to draw from each distribution.
        :return: DataFrame of posterior probability samples with sample number
                 index and prior categories as columns.
        """
        posterior_samples = {}
        for like_name, like in likelihood.items():
            posteriors = BayesRule._posterior__p_d__l_d(
                prior=prior, likelihood=like,
                num_samples=num_samples
            )
            for prior_name in posteriors.columns:
                posterior_samples[
                    like_name, prior_name
                ] = posteriors[prior_name]
        data = DataFrame(posterior_samples)
        data.columns.names = ['likelihood', 'prior']
        data.index.name = 'sample_index'
        return data

    def posterior(
            self, num_samples: Optional[int] = None
    ) -> Union[float, Series, DataFrame]:
        """
        Return samples from the posterior P(A|B).
        Columns are tuples of (a, b).
        """
        if isinstance(self._prior, float):
            if isinstance(self._likelihood, float):
                return BayesRule._posterior__p_f__l_f(
                    prior=self._prior,
                    likelihood=self._likelihood
                )
            elif is_any_float_map(self._likelihood):
                return BayesRule._posterior__p_f__l_fm(
                    prior=self._prior,
                    likelihood=self._likelihood
                )
            elif isinstance(self._likelihood, Beta):
                return BayesRule._posterior__p_f__l_b(
                    prior=self._prior,
                    likelihood=self._likelihood,
                    num_samples=num_samples
                )
            elif is_any_beta_map(self._likelihood):
                return BayesRule._posterior__p_f__l_bm(
                    prior=self._prior,
                    likelihood=self._likelihood,
                    num_samples=num_samples
                )
        elif isinstance(self._prior, Beta):
            if isinstance(self._likelihood, float):
                return BayesRule._posterior__p_b__l_f(
                    prior=self._prior,
                    likelihood=self._likelihood,
                    num_samples=num_samples
                )
            elif is_any_float_map(self._likelihood):
                return BayesRule._posterior__p_b__l_fm(
                    prior=self._prior,
                    likelihood=self._likelihood,
                    num_samples=num_samples
                )
            elif isinstance(self._likelihood, Beta):
                return BayesRule._posterior__p_b__l_b(
                    prior=self._prior,
                    likelihood=self._likelihood,
                    num_samples=num_samples
                )
            elif is_any_beta_map(self._likelihood):
                return BayesRule._posterior__p_b__l_bm(
                    prior=self._prior,
                    likelihood=self._likelihood,
                    num_samples=num_samples
                )
        elif is_any_float_map(self._prior):
            if is_any_float_map(self._likelihood):
                return BayesRule._posterior__p_fm__l__fm(
                    prior=self._prior,
                    likelihood=self._likelihood,
                )
        elif isinstance(self._prior, Dirichlet):
            if isinstance(self._likelihood, Dirichlet):
                return BayesRule._posterior__p_d__l_d(
                    prior=self._prior,
                    likelihood=self._likelihood,
                    num_samples=num_samples
                )
            elif is_any_dirichlet_map(self._likelihood):
                return BayesRule._posterior__p_d__l_dm(
                    prior=self._prior,
                    likelihood=self._likelihood,
                    num_samples=num_samples
                )
        else:
            raise TypeError(
                'wrong combination of types for prior and likelihood'
            )
