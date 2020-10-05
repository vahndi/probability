from itertools import product

from pandas import DataFrame, Series

from probability.distributions.mixins.rv_mixins import NUM_SAMPLES_COMPARISON
from tests.test_calculations.base_test import BaseTest


class TestMultiplyDistributions(BaseTest):

    def test_float___mul__rvs1d_name(self):

        self.assertEqual('0.5 * Beta(α=700, β=300)',
                         (0.5 * self.b1).name)

    def test_float__mul__rvs1d_result(self):

        expected = 0.5 * self.b1.rvs(100_000)
        actual = (0.5 * self.b1).output()
        self.assertAlmostEqual(expected.mean(), actual.mean(), 3)
        self.assertAlmostEqual(expected.std(), actual.std(), 3)
        self.assertIsInstance(actual, Series)
        self.assertEqual('0.5 * Beta(α=700, β=300)', actual.name)

    def test_rvs1d__mul__float_name(self):

        self.assertEqual('Beta(α=700, β=300) * 0.5',
                         (self.b1 * 0.5).name)

    def test_rvs1d__mul__float_result(self):

        expected = self.b1.rvs(100_000) * 0.5
        actual = (self.b1 * 0.5).output()
        self.assertAlmostEqual(expected.mean(), actual.mean(), 3)
        self.assertAlmostEqual(expected.std(), actual.std(), 3)
        self.assertIsInstance(actual, Series)
        self.assertEqual('Beta(α=700, β=300) * 0.5', actual.name)

    def test_float__mul__rvs2d_name(self):

        self.assertEqual('0.5 * Dirichlet(α1=500, α2=300, α3=200)',
                         (0.5 * self.d1).name)

    def test_float__mul__rvs2d_result(self):

        expected = (0.5 * self.d1.rvs(100_000, full_name=True)).rename(
            columns=lambda c: f'0.5 * {c}'
        )
        actual = (0.5 * self.d1).output()
        for name in self.d1.names:
            self.assertAlmostEqual(
                expected.mean()[f'0.5 * {str(self.d1)}[{name}]'],
                actual.mean()[f'0.5 * {str(self.d1)}[{name}]'], 3
            )
            self.assertAlmostEqual(
                expected.std()[f'0.5 * {str(self.d1)}[{name}]'],
                actual.std()[f'0.5 * {str(self.d1)}[{name}]'], 3
            )
        self.assertIsInstance(actual, DataFrame)

    def test_rvs2d__mul__float_name(self):

        self.assertEqual(f'{str(self.d1)} * 0.5',
                         (self.d1 * 0.5).name)

    def test_rvs2d__mul__float_result(self):

        expected = (self.d1.rvs(100_000, full_name=True) * 0.5).rename(
            columns=lambda c: f'{c} * 0.5'
        )
        actual = (self.d1 * 0.5).output()
        for name in self.d1.names:
            self.assertAlmostEqual(
                expected.mean()[f'{str(self.d1)}[{name}] * 0.5'],
                actual.mean()[f'{str(self.d1)}[{name}] * 0.5'], 3
            )
            self.assertAlmostEqual(
                expected.std()[f'{str(self.d1)}[{name}] * 0.5'],
                actual.std()[f'{str(self.d1)}[{name}] * 0.5'], 3
            )
        self.assertIsInstance(actual, DataFrame)

    def test_rvs1d__mul__rvs1d_name(self):

        self.assertEqual('Beta(α=700, β=300) * Beta(α=600, β=400)',
                         self.b1__mul__b2.name)

    def test_rvs1d__mul__rvs1d_result(self):

        result = self.b1__mul__b2.output()
        self.assertAlmostEqual(0.42, result.mean(), 2)
        self.assertAlmostEqual(0.014, result.std(), 3)
        self.assertEqual('Beta(α=700, β=300) * Beta(α=600, β=400)',
                         result.name)

    def test_float__mul__rvs1d__mul__rvs1d_name(self):

        self.assertEqual(
            '(0.5 * Beta(α=700, β=300)) * Beta(α=600, β=400)',
            (0.5 * self.b1 * self.b2).name
        )

    def test_float__mul__rvs1d__mul__rvs1d_result(self):

        expected = 0.5 * self.b1.rvs(100_000) * self.b2.rvs(100_000)
        actual = (0.5 * self.b1 * self.b2).output()
        self.assertAlmostEqual(expected.mean(), actual.mean(), 3)
        self.assertAlmostEqual(expected.std(), actual.std(), 3)
        self.assertEqual('(0.5 * Beta(α=700, β=300)) * Beta(α=600, β=400)',
                         actual.name)

    def test_rvs1d__mul__float__mul__rvs1d_name(self):

        self.assertEqual(
            '(Beta(α=700, β=300) * 0.5) * Beta(α=600, β=400)',
            (self.b1 * 0.5 * self.b2).name
        )

    def test_rvs1d__mul__float__mul__rvs1d_result(self):

        expected = self.b1.rvs(100_000) * 0.5 * self.b2.rvs(100_000)
        actual = (self.b1 * 0.5 * self.b2).output()
        self.assertAlmostEqual(expected.mean(), actual.mean(), 3)
        self.assertAlmostEqual(expected.std(), actual.std(), 3)
        self.assertEqual('(Beta(α=700, β=300) * 0.5) * Beta(α=600, β=400)',
                         actual.name)

    def test_rvs1d__mul__rvs1d__mul__float_name(self):

        self.assertEqual(
            '(Beta(α=700, β=300) * Beta(α=600, β=400)) * 0.5',
            (self.b1 * self.b2 * 0.5).name
        )

    def test_rvs1d__mul__rvs1d__mul__float_result(self):

        expected = self.b1.rvs(100_000) * self.b2.rvs(100_000) * 0.5
        actual = (self.b1 * self.b2 * 0.5).output()
        self.assertAlmostEqual(expected.mean(), actual.mean(), 3)
        self.assertAlmostEqual(expected.std(), actual.std(), 3)
        self.assertEqual('(Beta(α=700, β=300) * Beta(α=600, β=400)) * 0.5',
                         actual.name)

    def test_rvs1d__mul__rvs1d__mul__rvs1d_name(self):

        self.assertEqual(
            'Beta(α=500, β=500) * (Beta(α=700, β=300) * Beta(α=600, β=400))',
            self.b3__mul__b1__mul__b2.name
        )

    def test_rvs1d__mul__rvs1d__mul__rvs1d_result(self):

        result = self.b3__mul__b1__mul__b2.output()
        self.assertAlmostEqual(0.21, result.mean(), 2)
        self.assertAlmostEqual(0.0096, result.std(), 3)
        self.assertEqual(
            'Beta(α=500, β=500) * (Beta(α=700, β=300) * Beta(α=600, β=400))',
            result.name
        )

    def test_rvs1d__mul__rvs2d_name(self):

        self.assertEqual(
            f'{str(self.b1)} * {str(self.d1)}',
            (self.b1 * self.d1).name
        )

    def test_rvs1d__mul__rvs2d_result(self):

        actual = (self.b1 * self.d1).output()
        expected = (
            self.d1.rvs(100_000, full_name=True).mul(
                self.b1.rvs(100_000), axis=0
            )
        ).rename(
            columns=lambda c: f'{str(self.b1)} * {c}'
        )
        for name in self.d1.names:
            self.assertAlmostEqual(
                expected.mean()[f'{str(self.b1)} * {str(self.d1)}[{name}]'],
                actual.mean()[f'{str(self.b1)} * {str(self.d1)}[{name}]'], 3
            )
            self.assertAlmostEqual(
                expected.std()[f'{str(self.b1)} * {str(self.d1)}[{name}]'],
                actual.std()[f'{str(self.b1)} * {str(self.d1)}[{name}]'], 3
            )
        self.assertIsInstance(actual, DataFrame)

    def test_rvs2d__mul__rvs1d_name(self):

        self.assertEqual(
            f'{str(self.d1)} * {str(self.b1)}',
            (self.d1 * self.b1).name
        )

    def test_rvs2d__mul__rvs1d_result(self):

        actual = (self.d1 * self.b1).output()
        expected = (
            self.d1.rvs(100_000, full_name=True).mul(
                self.b1.rvs(100_000), axis=0
            )
        ).rename(
            columns=lambda c: f'{c} * {str(self.b1)}'
        )
        for name in self.d1.names:
            self.assertAlmostEqual(
                expected.mean()[f'{str(self.d1)}[{name}] * {str(self.b1)}'],
                actual.mean()[f'{str(self.d1)}[{name}] * {str(self.b1)}'], 3
            )
            self.assertAlmostEqual(
                expected.std()[f'{str(self.d1)}[{name}] * {str(self.b1)}'],
                actual.std()[f'{str(self.d1)}[{name}] * {str(self.b1)}'], 3
            )
        self.assertIsInstance(actual, DataFrame)

    def test_rvs2d__mul__rvs2d_name(self):

        self.assertEqual(
            f'{str(self.d1)} * {str(self.d2)}',
            (self.d1 * self.d2).name
        )

    def test_rvs2d__mul__rvs2d_result(self):

        actual = (self.d1 * self.d2).output()
        d1_samples = self.d1.rvs(NUM_SAMPLES_COMPARISON, full_name=True)
        d2_samples = self.d2.rvs(NUM_SAMPLES_COMPARISON, full_name=True)
        expected = DataFrame.from_dict({
            f'{col_1} * {col_2}':
                d1_samples[col_1] * d2_samples[col_2]
            for col_1, col_2 in zip(d1_samples.columns, d2_samples.columns)
        })
        for column in expected.columns:
            self.assertAlmostEqual(
                expected.mean()[column],
                actual.mean()[column], 3
            )
            self.assertAlmostEqual(
                expected.std()[column],
                actual.std()[column], 3
            )

    def test_rvs1d__mul__series_names(self):

        actual = self.b1 * self.b_series
        for ix, value in actual.iteritems():
            self.assertEqual(
                f'{str(self.b1)} * {str(self.b_series[ix])}',
                actual[ix].name
            )

    def test_rvs1d__mul__series_results(self):

        actual = self.b1 * self.b_series
        for ix, value in actual.iteritems():
            expected_output = (self.b1 * self.b_series[ix]).output()
            value_output = value.output()
            self.assertAlmostEqual(expected_output.mean(),
                                   value_output.mean(), 3)
            self.assertAlmostEqual(expected_output.std(),
                                   value_output.std(), 2)

    def test_series__mul__rvs1d__names(self):

        actual = self.b_series * self.b1
        for ix, value in actual.iteritems():
            self.assertEqual(
                f'{str(self.b_series[ix])} * {str(self.b1)}',
                actual[ix].name
            )

    def test_series__mul__rvs1d_results(self):

        actual = self.b_series * self.b1
        for ix, value in actual.iteritems():
            expected_output = (self.b_series[ix] * self.b1).output()
            actual_output = value.output()
            self.assertAlmostEqual(expected_output.mean(),
                                   actual_output.mean(), 3)
            self.assertAlmostEqual(expected_output.std(),
                                   actual_output.std(), 2)

    def test_rvs1d__mul__dataframe_names(self):

        actual = self.b1 * self.b_frame
        for ix, column in product(actual.index, actual.columns):
            self.assertEqual(
                f'{str(self.b1)} * {str(self.b_frame.loc[ix, column])}',
                actual.loc[ix, column].name
            )

    def test_rvs1d__mul__dataframe_results(self):

        actual = self.b1 * self.b_frame
        for ix, column in product(actual.index, actual.columns):
            expected_output = (self.b1 * self.b_frame.loc[ix, column]).output()
            actual_output = actual.loc[ix, column].output()
            self.assertAlmostEqual(expected_output.mean(),
                                   actual_output.mean(), 3)
            self.assertAlmostEqual(expected_output.std(),
                                   actual_output.std(), 2)

    def test_dataframe__mul__rvs1d_names(self):

        actual = self.b_frame * self.b1
        for ix, column in product(actual.index, actual.columns):
            self.assertEqual(
                f'{str(self.b_frame.loc[ix, column])} * {str(self.b1)}',
                actual.loc[ix, column].name
            )

    def test_dataframe__mul__rvs1d_results(self):

        actual = self.b_frame * self.b1
        for ix, column in product(actual.index, actual.columns):
            expected_output = (self.b_frame.loc[ix, column] * self.b1).output()
            actual_output = actual.loc[ix, column].output()
            self.assertAlmostEqual(expected_output.mean(),
                                   actual_output.mean(), 3)
            self.assertAlmostEqual(expected_output.std(),
                                   actual_output.std(), 2)
