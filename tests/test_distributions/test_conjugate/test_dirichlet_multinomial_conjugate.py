from unittest.case import TestCase

from pandas import DataFrame, Series

from probability.distributions import Dirichlet
from probability.distributions.conjugate.dirichlet_multinomial_conjugate import \
    DirichletMultinomialConjugate


class TestDirichletMultinomialConjugate(TestCase):

    def setUp(self) -> None:

        self.series = Series(data=['a'] * 5 + ['b'] * 3 + ['c'] * 2)
        self.multinomial_data = DataFrame({
            'a': [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2],
            'b': [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
            'e': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
            'f': [2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3, 2, 2, 2, 3]
        })

    def test_infer_posterior(self):

        expected = Dirichlet(alpha={'a': 5 + 1, 'b': 3 + 1, 'c': 2 + 1})
        actual = DirichletMultinomialConjugate.infer_posterior(self.series)
        self.assertEqual(expected, actual)

    def test_infer_posteriors(self):

        expected = DataFrame(
            data=[
                (1, 1, 'e', Dirichlet({1: 3, 2: 2, 3: 2})),
                (2, 1, 'e', Dirichlet({1: 2, 2: 3, 3: 2})),
                (1, 2, 'e', Dirichlet({1: 2, 2: 2, 3: 3})),
                (2, 2, 'e', Dirichlet({1: 3, 2: 2, 3: 2})),
                (1, 1, 'f', Dirichlet({2: 4, 3: 2})),
                (2, 1, 'f', Dirichlet({2: 3, 3: 3})),
                (1, 2, 'f', Dirichlet({2: 2, 3: 4})),
                (2, 2, 'f', Dirichlet({2: 4, 3: 2}))
            ],
            columns=['a', 'b', 'prob_var', 'Dirichlet']
        )
        actual = DirichletMultinomialConjugate.infer_posteriors(
            data=self.multinomial_data,
            prob_vars=['e', 'f'],
            cond_vars=['a', 'b']
        )
        for _, row in expected.iterrows():
            actual_dirichlet = actual.loc[
                (actual['a'] == row['a']) &
                (actual['b'] == row['b']) &
                (actual['prob_var'] == row['prob_var']),
                'Dirichlet'
            ].iloc[0]
            self.assertTrue(row['Dirichlet'] == actual_dirichlet)
