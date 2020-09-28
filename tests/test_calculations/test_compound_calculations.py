from probability.calculations.context import sync_context
from probability.distributions.mixins.rv_mixins import NUM_SAMPLES_COMPARISON
from tests.test_calculations.base_test import BaseTest


class TestCompoundCalculations(BaseTest):

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

    def test_sum_product__rvs1d_name(self):

        result = sum([(self.b1 * self.b2), ((1 - self.b1) * (1 - self.b2))])
        print(result.name)
        self.assertEqual(
            f'({str(self.b1)} * {str(self.b2)}) + '
            f'((1 - {str(self.b1)}) * (1 - {str(self.b2)}))',
            result.name
        )

    def test_sum_product__rvs1d_result(self):

        result = sum([(self.b1 * self.b2), (1 - self.b1) * (1 - self.b2)])
        sync_context(result)
        b1s = self.b1.rvs(NUM_SAMPLES_COMPARISON)
        b2s = self.b2.rvs(NUM_SAMPLES_COMPARISON)
        b1comps = 1 - b1s
        b2comps = 1 - b2s
        expected = (b1s * b2s) + (b1comps * b2comps)
        actual = result.output()
        self.assertAlmostEqual(expected.mean(), actual.mean(), 4)
        self.assertAlmostEqual(expected.std(), actual.std(), 4)
