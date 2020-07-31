from unittest.case import TestCase

from pandas import Series, DataFrame

from probability.distributions import Beta


class TestBeta(TestCase):

    def setUp(self) -> None:

        self.series = Series(data=[0] * 6 + [1] * 4)
        self.data_frame = DataFrame({
            'a': [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2],
            'b': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            'c': [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1],
            'd': [1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
        })

    def test_from_series(self):

        expected = Beta(alpha=4, beta=6)
        actual = Beta.fit(self.series)
        self.assertEqual(expected.alpha, actual.alpha)
        self.assertEqual(expected.beta, actual.beta)

    def test_from_data_frame(self):

        expected = DataFrame([
            {'a': 1, 'b': 1, 'p': 'c', 'Beta': Beta(0, 3)},
            {'a': 2, 'b': 1, 'p': 'c', 'Beta': Beta(1, 2)},
            {'a': 1, 'b': 2, 'p': 'c', 'Beta': Beta(2, 1)},
            {'a': 2, 'b': 2, 'p': 'c', 'Beta': Beta(3, 0)},
            {'a': 1, 'b': 1, 'p': 'd', 'Beta': Beta(3, 0)},
            {'a': 2, 'b': 1, 'p': 'd', 'Beta': Beta(2, 1)},
            {'a': 1, 'b': 2, 'p': 'd', 'Beta': Beta(1, 2)},
            {'a': 2, 'b': 2, 'p': 'd', 'Beta': Beta(0, 3)}
        ])
        actual = Beta.fits(
            data=self.data_frame,
            prob_vars=['c', 'd'],
            cond_vars=['a', 'b']
        )
        for _, row in expected.iterrows():
            actual_beta = actual.loc[
                (actual['a'] == row['a']) &
                (actual['b'] == row['b']) &
                (actual['p'] == row['p']),
                'Beta'
            ].iloc[0]
            self.assertEqual(row['Beta'].alpha, actual_beta.alpha)
            self.assertEqual(row['Beta'].beta, actual_beta.beta)
