from typing import Any
from pandas import DataFrame

from probability.distributions.mixins.rv_mixins import NUM_SAMPLES_COMPARISON
from tests.test_calculations.base_test import BaseTest


class TestAddDistributions(BaseTest):

    def test_rvs1d__add__rvs1d_name(self):

        result = self.b1 + self.b2
        self.assertEqual(f'{str(self.b1)} + {str(self.b2)}',
                         result.name)

    def test_rvs1d__add__rvs1d_result(self):

        actual = (self.b1 + self.b2).output()
        expected = (self.b1.rvs(NUM_SAMPLES_COMPARISON) +
                    self.b2.rvs(NUM_SAMPLES_COMPARISON))
        self.assertAlmostEqual(expected.mean(), actual.mean(), 3)
        self.assertAlmostEqual(expected.std(), actual.std(), 2)

    def test_rvs2d__add__rvs2d_name(self):

        result = self.d1 + self.d2
        self.assertEqual(f'{str(self.d1)} + {str(self.d2)}',
                         result.name)

    def test_rvs2d__add__rvs2d_result(self):

        actual = (self.d1 + self.d2).output()
        d1_samples = self.d1.rvs(NUM_SAMPLES_COMPARISON, full_name=True)
        d2_samples = self.d2.rvs(NUM_SAMPLES_COMPARISON, full_name=True)
        expected = DataFrame.from_dict({
            f'{col_1} + {col_2}':
                d1_samples[col_1] + d2_samples[col_2]
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

    def test_sum__rvs1d_rvs1d_name(self):

        sum_calc: Any = sum([self.b1, self.b2])
        result = sum_calc.output()
        self.assertEqual(f'{str(self.b1)} + {str(self.b2)}',
                         result.name)

    def test_sum__rvs1d_rvs1d_result(self):

        sum_calc: Any = sum([self.b1, self.b2])
        actual = sum_calc.output()
        expected = (self.b1.rvs(NUM_SAMPLES_COMPARISON) +
                    self.b2.rvs(NUM_SAMPLES_COMPARISON))
        self.assertAlmostEqual(expected.mean(), actual.mean(), 3)
        self.assertAlmostEqual(expected.std(), actual.std(), 2)

    def test_sum__rvs2d_rvs2d_name(self):

        result: Any = sum([self.d1, self.d2])
        self.assertEqual(f'{str(self.d1)} + {str(self.d2)}',
                         result.name)

    def test_sum__rvs2d_rvs2d_result(self):

        sum_calc: Any = sum([self.d1, self.d2])
        actual = sum_calc.output()
        d1_samples = self.d1.rvs(NUM_SAMPLES_COMPARISON, full_name=True)
        d2_samples = self.d2.rvs(NUM_SAMPLES_COMPARISON, full_name=True)
        expected = DataFrame.from_dict({
            f'{col_1} + {col_2}':
                d1_samples[col_1] + d2_samples[col_2]
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

    def test_rvs1d__add__int_scalar(self):

        calc = self.b1 + 2
        self.assertEqual(f'{str(self.b1)} + 2', calc.name)
        self.assertAlmostEqual(2.7, calc.output().mean(), 2)

    def test_rvs1d__add__series(self):

        actual = self.b1 + self.b_series
        for ix, calc in actual.items():
            self.assertEqual(
                f'{str(self.b1)} + {str(self.b_series[ix])}',
                calc.name
            )

    def test_rvs1d__sub__rvs1d(self):

        calc = self.b1 - self.b2
        self.assertEqual(f'{str(self.b1)} - {str(self.b2)}', calc.name)
        self.assertAlmostEqual(0.1, calc.output().mean(), 2)

