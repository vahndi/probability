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
