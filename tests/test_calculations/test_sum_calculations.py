from pandas import DataFrame, Series

from probability.calculations.context import sync_context
from probability.distributions.mixins.rv_mixins import NUM_SAMPLES_COMPARISON
from tests.test_calculations.base_test import BaseTest


class TestSumCalculations(BaseTest):

    def test_sum__product_name(self):

        product = self.d1 * self.d2
        sum_product = product.sum()
        self.assertEqual(
            f'sum({str(self.d1)} * {str(self.d2)})',
            sum_product.name
        )

    def test_sum__product_result(self):

        product = self.d1 * self.d2
        sum_product = product.sum()
        sync_context(sum_product)
        output = sum_product.output(NUM_SAMPLES_COMPARISON)
        self.assertIsInstance(output, Series)
        d1_out = sum_product.context[str(self.d1)]
        d2_out = sum_product.context[str(self.d2)]
        d1_out.columns = ['a', 'b', 'c']
        d2_out.columns = ['a', 'b', 'c']
        d1_d2 = (d1_out * d2_out)
        d1_d2_sum = d1_d2.sum(axis=1)
        self.assertTrue(output.equals(d1_d2_sum))

    def test_bayes_rule_dirichlet_name(self):

        product = self.d1 * self.d2
        sum_product = product.sum()
        posterior = product / sum_product
        sync_context(posterior)
        output = posterior.output(NUM_SAMPLES_COMPARISON)
        self.assertIsInstance(output, DataFrame)
        d1_out = sum_product.context[str(self.d1)]
        d2_out = sum_product.context[str(self.d2)]
        d1_out.columns = ['a', 'b', 'c']
        d2_out.columns = ['a', 'b', 'c']
        d1_d2 = (d1_out * d2_out)
        d1_d2_sum = d1_d2.sum(axis=1)
        result = d1_d2.div(d1_d2_sum, axis=0)
        output.columns = ['a', 'b', 'c']
        self.assertTrue(output.equals(result))
