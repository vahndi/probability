from unittest.case import TestCase

from pandas import Series, DataFrame

from probability.distributions import Beta, BetaBinomialConjugate


class TestBetaBinomialConjugate(TestCase):

    def setUp(self) -> None:

        self.series = Series(data=[0] * 6 + [1] * 4)
        self.binomial_data = DataFrame({
            'a': [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2],
            'b': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            'c': [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1],
            'd': [1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
        })
        self.multinomial_data = DataFrame({
            'a': [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2],
            'b': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            'e': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        })

    def test_infer_posterior(self):

        expected = Beta(alpha=1 + 4, beta=1 + 6)
        actual = BetaBinomialConjugate.infer_posterior(self.series)
        self.assertEqual(expected, actual)

    def test_infer_posteriors(self):

        b__0_3 = Beta(1 + 0, 1 + 3)
        b__1_2 = Beta(1 + 1, 1 + 2)
        b__2_1 = Beta(1 + 2, 1 + 1)
        b__3_0 = Beta(1 + 3, 1 + 0)

        expected = DataFrame(
            data=[
                (1, 1, 'c', 1, b__0_3),
                (2, 1, 'c', 1, b__1_2),
                (1, 2, 'c', 1, b__2_1),
                (2, 2, 'c', 1, b__3_0),
                (1, 1, 'd', 1, b__3_0),
                (2, 1, 'd', 1, b__2_1),
                (1, 2, 'd', 1, b__1_2),
                (2, 2, 'd', 1, b__0_3)
            ],
            columns=['a', 'b', 'prob_var', 'prob_val', 'Beta']
        )
        actual = BetaBinomialConjugate.infer_posteriors(
            data=self.binomial_data,
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

    def test_infer_posteriors_with_stats(self):

        b__0_3 = Beta(1 + 0, 1 + 3)
        b__1_2 = Beta(1 + 1, 1 + 2)
        b__2_1 = Beta(1 + 2, 1 + 1)
        b__3_0 = Beta(1 + 3, 1 + 0)

        expected = DataFrame(
            data=[
                (1, 1, 'c', 1, b__0_3, b__0_3.mean(), b__0_3.interval(.95)),
                (2, 1, 'c', 1, b__1_2, b__1_2.mean(), b__1_2.interval(.95)),
                (1, 2, 'c', 1, b__2_1, b__2_1.mean(), b__2_1.interval(.95)),
                (2, 2, 'c', 1, b__3_0, b__3_0.mean(), b__3_0.interval(.95)),
                (1, 1, 'd', 1, b__3_0, b__3_0.mean(), b__3_0.interval(.95)),
                (2, 1, 'd', 1, b__2_1, b__2_1.mean(), b__2_1.interval(.95)),
                (1, 2, 'd', 1, b__1_2, b__1_2.mean(), b__1_2.interval(.95)),
                (2, 2, 'd', 1, b__0_3, b__0_3.mean(), b__0_3.interval(.95))
            ],
            columns=['a', 'b', 'prob_var', 'prob_val', 'Beta',
                     'mean', 'interval__0.95']
        )
        actual = BetaBinomialConjugate.infer_posteriors(
            data=self.binomial_data,
            prob_vars=['c', 'd'],
            cond_vars=['a', 'b'],
            stats=['mean', {'interval': 0.95}]
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
