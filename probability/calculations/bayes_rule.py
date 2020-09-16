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

        :param prior: Beta or Dirichlet where each dimension represents
                      one likelihood category.
        :param likelihood: If prior is Beta, Series with values of float or
                           Beta likelihoods.
                           if prior is Dirichlet, DataFrame with columns of
                           float likelihoods or Series with values of Dirichlet
                           likelihoods

                           (b) Beta
                           (c) Mapping[Any, Single figure]
                           (d) Mapping[Any, Beta]

        N.B. need to implement Dirichlet because of evidence calculation
        e.g. if prior_1 = 0.3, prior_2 = 0.5 and prior_3 = 0.2 and
                like_1 = 0.7, like_2 = 0.2 and like_1 = 0.1
        Then for p1:
            using Beta then evidence is
                (0.3 * 0.7) + (0.5 + 0.2) * (0.2 + 0.1) = 0.42
            using Dirichlet evidence is
                (0.3 * 0.7) + (0.5 * 0.2) + (0.2 * 0.1) = 0.33
        This would lead to a lower overall probability using Beta than reality,
        in this case.
        """
        self._prior: Dirichlet = prior
        self._likelihood: Mapping[Any, Dirichlet] = likelihood

    # @staticmethod
    # def from_counts(
    #         counts: DataFrame,
    #         likelihood_uncertainty: bool = False,
    # ) -> Dict[str, 'BayesRule']:
    #     """
    #     Return a new BayesRule class using a DataFrame of counts.
    #
    #     :param counts: DataFrame of counts where the index represents the
    #                    different states of the evidence B, and each column
    #                    represents the likelihood for one value of the prior A.
    #     :param likelihood_uncertainty: Set to True if there is uncertainty
    #                                    around the likelihood i.e. if a
    #                                    likelihood value could be in any other
    #                                    category.
    #     """
    #     bayes_rules = {}
    #
    #     n = counts.sum().sum()
    #
    #     for prior_state in counts.columns:
    #         prior_counts = counts[prior_state]
    #         n_prior = prior_counts.sum()
    #         prior = BetaBinomialConjugate(1, 1, n, n_prior).posterior()
    #         if likelihood_uncertainty:
    #             likelihood = Series(
    #                 index=counts.index,
    #                 data=prior_counts.map(
    #                     lambda k_likelihood: BetaBinomialConjugate(
    #                         1, 1, n_prior, k_likelihood
    #                     ).posterior()
    #                 )
    #             )
    #         else:
    #             likelihood = Series(
    #                 index=counts.index,
    #                 data=prior_counts.map(
    #                     lambda k_likelihood: BetaBinomialConjugate(
    #                         1, 1, k_likelihood, k_likelihood
    #                     ).posterior()
    #                 )
    #             )
    #
    #         bayes_rules[prior_state] = BayesRule(
    #             prior=prior,
    #             likelihood=likelihood,
    #         )
    #
    #     return bayes_rules

    @staticmethod
    def from_counts(
            data: DataFrame,
    ) -> 'BayesRule':
        """
        Return a new BayesRule class using a DataFrame of counts.

        :param data: DataFrame of counts where the index represents the
                     different states of the evidence B, and each column
                     represents the likelihood for one value of the prior A.
        """
        prior = Dirichlet(1 + data.sum())
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
