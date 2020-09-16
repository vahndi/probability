from typing import Mapping, Any, Union

from pandas import DataFrame, Series

from probability.distributions import Dirichlet


class BayesRule(object):

    def __init__(
            self,
            prior: Dirichlet,
            likelihood: Union[Mapping[Any, Dirichlet], Series]
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
        self._prior: Dirichlet = prior
        self._likelihood: Mapping[Any, Dirichlet] = likelihood

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
    def prior(self) -> Dirichlet:

        return self._prior

    @property
    def likelihood(self) -> Mapping[Any, Dirichlet]:

        return self._likelihood

    def sample_posterior(
            self, num_samples: int
    ) -> DataFrame:
        """
        Return samples from the posterior P(A|B).
        Columns are tuples of (a, b).
        """
        posterior_samples = {}
        for like_name in self._likelihood.keys():
            p_prior = self._prior.rvs(num_samples)
            p_likelihood = self._likelihood[like_name].rvs(num_samples)
            p_evidence = (p_prior * p_likelihood).sum(axis=1)
            posterior = (p_prior * p_likelihood).div(p_evidence, axis=0)
            for prior_name in posterior.columns:
                posterior_samples[like_name, prior_name] = posterior[prior_name]
        data = DataFrame(posterior_samples)
        data.columns.names = ['likelihood', 'prior']
        data.index.name = 'sample_index'
        return data
