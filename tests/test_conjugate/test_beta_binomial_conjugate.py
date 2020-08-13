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

        expected = Beta(alpha=4, beta=6)
        actual = BetaBinomialConjugate.infer_posterior(self.series)
        self.assertEqual(expected, actual)

    def test_infer_posteriors_binomial(self):

        expected = DataFrame(
            data=[
                (1, 1, 'c', 1, Beta(0, 3)),
                (2, 1, 'c', 1, Beta(1, 2)),
                (1, 2, 'c', 1, Beta(2, 1)),
                (2, 2, 'c', 1, Beta(3, 0)),
                (1, 1, 'd', 1, Beta(3, 0)),
                (2, 1, 'd', 1, Beta(2, 1)),
                (1, 2, 'd', 1, Beta(1, 2)),
                (2, 2, 'd', 1, Beta(0, 3))
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

    def test_infer_posteriors_binomial_with_stats(self):

        expected = DataFrame(
            data=[
                (1, 1, 'c', 1, Beta(0, 3), Beta(0, 3).mean(),
                 Beta(0, 3).interval(.95)),
                (2, 1, 'c', 1, Beta(1, 2), Beta(1, 2).mean(),
                 Beta(1, 2).interval(.95)),
                (1, 2, 'c', 1, Beta(2, 1), Beta(2, 1).mean(),
                 Beta(2, 1).interval(.95)),
                (2, 2, 'c', 1, Beta(3, 0), Beta(3, 0).mean(),
                 Beta(3, 0).interval(.95)),
                (1, 1, 'd', 1, Beta(3, 0), Beta(3, 0).mean(),
                 Beta(3, 0).interval(.95)),
                (2, 1, 'd', 1, Beta(2, 1), Beta(2, 1).mean(),
                 Beta(2, 1).interval(.95)),
                (1, 2, 'd', 1, Beta(1, 2), Beta(1, 2).mean(),
                 Beta(1, 2).interval(.95)),
                (2, 2, 'd', 1, Beta(0, 3), Beta(0, 3).mean(),
                 Beta(0, 3).interval(.95))
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

    def test_infer_posteriors_multinomial(self):

        expected = DataFrame(
            data=[
                (1, 1, 'e', 1, Beta(3, 0)),
                (2, 1, 'e', 1, Beta(1, 2)),
                (1, 2, 'e', 1, Beta(0, 3)),
                (2, 2, 'e', 1, Beta(0, 3)),
                (1, 1, 'e', 2, Beta(0, 3)),
                (2, 1, 'e', 2, Beta(2, 1)),
                (1, 2, 'e', 2, Beta(2, 1)),
                (2, 2, 'e', 2, Beta(0, 3)),
                (1, 1, 'e', 3, Beta(0, 3)),
                (2, 1, 'e', 3, Beta(0, 3)),
                (1, 2, 'e', 3, Beta(1, 2)),
                (2, 2, 'e', 3, Beta(3, 0)),
            ],
            columns=['a', 'b', 'prob_var', 'prob_val', 'Beta']
        )
        actual = BetaBinomialConjugate.infer_posteriors(
            data=self.multinomial_data,
            prob_vars='e',
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
