from unittest.case import TestCase

from pandas import Series

from probability.calculations.bayes_rule.multiple_bayes_rule import \
    MultipleBayesRule
from probability.distributions import Dirichlet


class TestMultipleBayesRule(TestCase):

    def setUp(self) -> None:

        self.prior_dirichlet = Dirichlet([1 + 70, 1 + 25, 1 + 5])
        self.likelihood_dirichlet = Dirichlet([1 + 6, 1 + 3, 1 + 1])
        self.likelihood_dirichlet_map = Series({
            '$100': Dirichlet([1 + 8, 1 + 1, 1 + 1]),
            '$200': Dirichlet([1 + 6, 1 + 3, 1 + 1])
        })
        self.prior_float_map = Series({'$100': 0.3, '$200': 0.2, '$300': 0.5})
        self.likelihood_float_map = Series({
            '$100': 0.1, '$200': 0.2, '$300': 0.3
        })

    def test_posterior__p_d__l_d(self):

        posterior = MultipleBayesRule(
            prior=self.prior_dirichlet,
            likelihood=self.likelihood_dirichlet
        ).posterior()
        self.assertEqual(
            f'({str(self.prior_dirichlet)} * {str(self.likelihood_dirichlet)})'
            f' / (sum'
            f'({str(self.prior_dirichlet)} * {str(self.likelihood_dirichlet)})'
            f')',
            posterior.name
        )

    def test_posterior__p_d__l_dm(self):

        posterior = MultipleBayesRule(
            prior=self.prior_dirichlet,
            likelihood=self.likelihood_dirichlet_map
        ).posterior()
        for key, likelihood in self.likelihood_dirichlet_map.items():
            self.assertEqual(
                f'({str(self.prior_dirichlet)} * {str(likelihood)})'
                f' / (sum'
                f'({str(self.prior_dirichlet)} * {str(likelihood)})'
                f')',
                posterior[key].name
            )

    def test_posterior__p_fm__l_fm(self):

        posterior = MultipleBayesRule(
            prior=self.prior_float_map,
            likelihood=self.likelihood_float_map
        ).posterior()
        normalization = (self.prior_float_map *
                         self.likelihood_float_map).sum()
        for key in self.prior_float_map.keys():
            self.assertEqual(
                self.prior_float_map[key] *
                self.likelihood_float_map[key] /
                normalization,
                posterior[key]
            )
