from pandas import DataFrame

from probability.distributions.mixins.rv_mixins import NUM_SAMPLES_COMPARISON
from tests.test_calculations.base_test import BaseTest


class TestDivideDistributions(BaseTest):

    def test_float__div__rvs1d_name(self):

        self.assertEqual(
            (1 / self.b1).name,
            f'1 / {str(self.b1)}'
        )

    def test_float__div__rvs1d_result(self):

        actual = (1 / self.b1).output()
        expected = 1 / self.b1.rvs(NUM_SAMPLES_COMPARISON)
        self.assertAlmostEqual(expected.mean(), actual.mean(), 3)
        self.assertAlmostEqual(expected.std(), actual.std(), 3)

    def test_float__div__rvs2d_name(self):

        self.assertEqual(
            (1 / self.d1).name,
            f'1 / {str(self.d1)}'
        )

    def test_float__div__rvs2d_result(self):

        actual = (1 / self.d1).output()
        expected = (
            1 / self.d1.rvs(NUM_SAMPLES_COMPARISON, full_name=True)
        ).rename(columns=lambda c: f'1 / {c}')
        for column in actual.columns:
            self.assertAlmostEqual(expected[column].mean(),
                                   actual[column].mean(), 2)
            self.assertAlmostEqual(expected[column].std(),
                                   actual[column].std(), 2)

    def test_rvs1d__div__float_name(self):

        self.assertEqual(
            (self.b1 / 2).name,
            f'{str(self.b1)} / 2'
        )

    def test_rvs1d__div__float_result(self):

        actual = (self.b1 / 2).output()
        expected = self.b1.rvs(NUM_SAMPLES_COMPARISON) / 2
        self.assertAlmostEqual(expected.mean(), actual.mean(), 3)
        self.assertAlmostEqual(expected.std(), actual.std(), 3)

    def test_rvs2d__div__float_name(self):

        self.assertEqual(
            (self.d1 / 2).name,
            f'{str(self.d1)} / 2'
        )

    def test_rvs2d__div__float_result(self):

        actual = (self.d1 / 2).output()
        expected = (
            self.d1.rvs(NUM_SAMPLES_COMPARISON, full_name=True) / 2
        ).rename(columns=lambda c: f'{c} / 2')
        for column in actual.columns:
            self.assertAlmostEqual(expected[column].mean(),
                                   actual[column].mean(), 3)
            self.assertAlmostEqual(expected[column].std(),
                                   actual[column].std(), 3)

    def test_rvs1d__div__rvs1d_name(self):

        self.assertEqual(
            (self.b1 / self.b2).name,
            (f'{str(self.b1)} / {str(self.b2)}')
        )

    def test_rvs1d__div__rvs1d_result(self):

        actual = (self.b1 / self.b2).output()
        expected = (
            self.b1.rvs(NUM_SAMPLES_COMPARISON) /
            self.b2.rvs(NUM_SAMPLES_COMPARISON)
        )
        self.assertAlmostEqual(expected.mean(), actual.mean(), 3)
        self.assertAlmostEqual(expected.std(), actual.std(), 3)

    def test_rvs2d__div__rvs2d_name(self):

        self.assertEqual(
            f'{str(self.d1)} / {str(self.d2)}',
            (self.d1 / self.d2).name
        )

    def test_rvs2d__div__rvs2d_result(self):

        actual = (self.d1 / self.d2).output()
        d1_samples = self.d1.rvs(NUM_SAMPLES_COMPARISON, full_name=True)
        d2_samples = self.d2.rvs(NUM_SAMPLES_COMPARISON, full_name=True)
        expected = DataFrame.from_dict({
            f'{col_1} / {col_2}':
                d1_samples[col_1] / d2_samples[col_2]
            for col_1, col_2 in zip(d1_samples.columns, d2_samples.columns)
        })
        for column in expected.columns:
            self.assertAlmostEqual(
                expected.mean()[column],
                actual.mean()[column], 2
            )
            self.assertAlmostEqual(
                expected.std()[column],
                actual.std()[column], 2
            )

    def test_rvs1d__div__rvs2d_name(self):

        self.assertEqual(
            f'{str(self.b1)} / {str(self.d1)}',
            (self.b1 / self.d1).name
        )

    def test_rvs1d__div__rvs2d_result(self):

        actual = (self.b1 / self.d1).output()
        expected = (
            self.d1.rvs(100_000, full_name=True).div(
                self.b1.rvs(100_000), axis=0
            )
        ).rename(
            columns=lambda c: f'{str(self.b1)} / {c}'
        )
        for name in self.d1.names:
            self.assertAlmostEqual(
                expected.mean()[f'{str(self.b1)} / {str(self.d1)}[{name}]'],
                actual.mean()[f'{str(self.b1)} / {str(self.d1)}[{name}]'], 3
            )
            self.assertAlmostEqual(
                expected.std()[f'{str(self.b1)} / {str(self.d1)}[{name}]'],
                actual.std()[f'{str(self.b1)} / {str(self.d1)}[{name}]'], 3
            )
        self.assertIsInstance(actual, DataFrame)

    def test_rvs2d__div__rvs1d_name(self):

        self.assertEqual(
            f'{str(self.d1)} / {str(self.b1)}',
            (self.d1 / self.b1).name
        )

    def test_rvs2d__div__rvs1d_result(self):

        actual = (self.d1 / self.b1).output()
        expected = (
            self.d1.rvs(100_000, full_name=True).div(
                self.b1.rvs(100_000), axis=0
            )
        ).rename(
            columns=lambda c: f'{c} / {str(self.b1)}'
        )
        for name in self.d1.names:
            self.assertAlmostEqual(
                expected.mean()[f'{str(self.d1)}[{name}] / {str(self.b1)}'],
                actual.mean()[f'{str(self.d1)}[{name}] / {str(self.b1)}'], 3
            )
            self.assertAlmostEqual(
                expected.std()[f'{str(self.d1)}[{name}] / {str(self.b1)}'],
                actual.std()[f'{str(self.d1)}[{name}] / {str(self.b1)}'], 3
            )
        self.assertIsInstance(actual, DataFrame)
