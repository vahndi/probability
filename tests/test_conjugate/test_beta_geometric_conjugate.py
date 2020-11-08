from unittest.case import TestCase

from pandas import Series, DataFrame

from probability.distributions import Beta
from probability.distributions.conjugate.beta_geometric_conjugate import \
    BetaGeometricConjugate


class TestBetaGeometricConjugate(TestCase):

    def setUp(self) -> None:

        self.series = Series([
            0, 0, 0, 1, 0, 0, 0, 0, 0, 1
        ])
        self.geometric_data = DataFrame({
            'a': [1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            'b': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            'c': [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            'd': [1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1],
        })

    def test_infer_posterior(self):

        expected = Beta(alpha=1 + 2, beta=1 + 10 - 2)
        actual = BetaGeometricConjugate.infer_posterior(self.series)
        self.assertEqual(expected, actual)

    def test_infer_posteriors(self):

        g__1_1 = Beta(1 + 1, 1 + 1)
        g__1_2 = Beta(1 + 1, 1 + 2)
        g__1_3 = Beta(1 + 1, 1 + 3)
        g__1_4 = Beta(1 + 1, 1 + 4)
        g__2_0 = Beta(1 + 2, 1 + 0)
        g__2_1 = Beta(1 + 2, 1 + 1)
        g__2_2 = Beta(1 + 2, 1 + 2)
        g__2_3 = Beta(1 + 2, 1 + 3)

        expected = DataFrame(
            data=[
                (1, 1, 'c', 1, g__1_1),
                (2, 1, 'c', 1, g__1_2),
                (1, 2, 'c', 1, g__1_3),
                (2, 2, 'c', 1, g__1_4),
                (1, 1, 'd', 1, g__2_0),
                (2, 1, 'd', 1, g__2_1),
                (1, 2, 'd', 1, g__2_2),
                (2, 2, 'd', 1, g__2_3)
            ],
            columns=['a', 'b', 'prob_var', 'prob_val', 'Beta']
        )
        actual = BetaGeometricConjugate.infer_posteriors(
            data=self.geometric_data,
            prob_vars=['c', 'd'],
            cond_vars=['a', 'b']
        )
        for _, row in expected.iterrows():
            actual_beta = actual.loc[
                (actual['a'] == row['a']) &
                (actual['b'] == row['b']) &
                (actual['prob_var'] == row['prob_var']) &
                (actual['prob_val'] == row['prob_val']),
                'Beta'
            ].iloc[0]
            self.assertTrue(row['Beta'] == actual_beta)
