from unittest.case import TestCase

from pandas import DataFrame, Series

from probability.distributions import Beta, Dirichlet
from probability.distributions.mixins.rv_mixins import NUM_SAMPLES_COMPARISON


class TestDistributionCalculation(TestCase):

    def setUp(self) -> None:

        self.b1 = Beta(700, 300)
        self.b2 = Beta(600, 400)
        self.b3 = Beta(500, 500)
        self.d1 = Dirichlet([500, 300, 200])
        self.b1__mul__b2 = self.b1 * self.b2
        self.b3__mul__b1__mul__b2 = self.b3 * self.b1__mul__b2
        self.b1__mul__comp__b1 = self.b1 * (1 - self.b1)

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
        dist_name = 'Dirichlet(α1=500, α2=300, α3=200)'
        for name in self.d1.names:
            self.assertAlmostEqual(
                expected.mean()[f'0.5 * {dist_name}[{name}]'],
                actual.mean()[f'0.5 * {dist_name}[{name}]'], 3
            )
            self.assertAlmostEqual(
                expected.std()[f'0.5 * {dist_name}[{name}]'],
                actual.std()[f'0.5 * {dist_name}[{name}]'], 3
            )
        self.assertIsInstance(actual, DataFrame)

    def test_rvs2d__mul__float_name(self):

        self.assertEqual('Dirichlet(α1=500, α2=300, α3=200) * 0.5',
                         (self.d1 * 0.5).name)

    def test_rvs2d__mul__float_result(self):

        expected = (self.d1.rvs(100_000, full_name=True) * 0.5).rename(
            columns=lambda c: f'{c} * 0.5'
        )
        actual = (self.d1 * 0.5).output()
        dist_name = 'Dirichlet(α1=500, α2=300, α3=200)'
        for name in self.d1.names:
            self.assertAlmostEqual(
                expected.mean()[f'{dist_name}[{name}] * 0.5'],
                actual.mean()[f'{dist_name}[{name}] * 0.5'], 3
            )
            self.assertAlmostEqual(
                expected.std()[f'{dist_name}[{name}] * 0.5'],
                actual.std()[f'{dist_name}[{name}] * 0.5'], 3
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

    def test_comp__rvs1d__name(self):

        result = 1 - self.b1
        self.assertEqual('1 - Beta(α=700, β=300)', result.name)

    def test_comp__rvs1d__result(self):

        result = (1 - self.b1).output()
        expected = 1 - (self.b1.rvs(NUM_SAMPLES_COMPARISON))
        self.assertAlmostEqual(expected.mean(), result.mean(), 3)
        self.assertAlmostEqual(expected.std(), result.std(), 2)
        self.assertEqual('1 - Beta(α=700, β=300)',
                         result.name)

    def test_rvs1d__mul__rvs1d__mul__rvs1d_name(self):

        self.assertEqual(
            'Beta(α=500, β=500) * (Beta(α=700, β=300) * Beta(α=600, β=400))',
            self.b3__mul__b1__mul__b2.name
        )

    def test_rvs1d__mul__rvs1d__mul__rvs1d_result(self):

        result = self.b3__mul__b1__mul__b2.output()
        self.assertAlmostEqual(0.21, result.mean(), 2)
        self.assertAlmostEqual(0.0096, result.std(), 4)
        self.assertEqual(
            'Beta(α=500, β=500) * (Beta(α=700, β=300) * Beta(α=600, β=400))',
            result.name
        )

    def test_rvs1d__mul__comp__rvs1d_name(self):

        self.assertEqual(
            'Beta(α=700, β=300) * (1 - Beta(α=700, β=300))',
            self.b1__mul__comp__b1.name
        )

    def test_rvs1d__mul__comp__rvs1d_result(self):

        result = self.b1__mul__comp__b1.output()
        self.assertAlmostEqual(0.21, result.mean(), 2)
        self.assertAlmostEqual(0.006, result.std(), 3)
        self.assertEqual('Beta(α=700, β=300) * (1 - Beta(α=700, β=300))',
                         result.name)
