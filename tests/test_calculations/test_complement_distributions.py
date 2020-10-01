from probability.distributions.mixins.rv_mixins import NUM_SAMPLES_COMPARISON
from tests.test_calculations.base_test import BaseTest


class TestComplementDistributions(BaseTest):

    def test_comp__rvs1d__name(self):

        result = 1 - self.b1
        self.assertEqual(f'1 - {str(self.b1)}', result.name)

    def test_comp__rvs1d__result(self):

        result = (1 - self.b1).output()
        expected = 1 - (self.b1.rvs(NUM_SAMPLES_COMPARISON))
        self.assertAlmostEqual(expected.mean(), result.mean(), 3)
        self.assertAlmostEqual(expected.std(), result.std(), 2)
        self.assertEqual(f'1 - {str(self.b1)}', result.name)

    def test_comp__rvs2d__name(self):

        result = 1 - self.d1
        self.assertEqual(f'1 - {str(self.d1)}', result.name)

    def test_comp__rvs2d__result(self):

        result = (1 - self.d1).output()
        expected = (
                1 - self.d1.rvs(NUM_SAMPLES_COMPARISON, full_name=True)
        ).rename(columns=lambda c: f'1 - {c}')
        for column in expected.columns:
            self.assertAlmostEqual(expected[column].mean(),
                                   result[column].mean(), 3)
            self.assertAlmostEqual(expected[column].std(),
                                   result[column].std(), 2)
