from unittest import TestCase

from probability.pandas.discrete_distribution import DiscreteDistribution
from tests.shared import read_distribution_data, series_are_equivalent


class TestDiscreteDistribution(TestCase):

    def test_product_rule(self):

        p_AB = DiscreteDistribution(read_distribution_data('P(A,B)'))
        # margins
        p_A = p_AB.margin('A')
        p_B = p_AB.margin('B')
        # conditions
        p_A__B = p_AB.condition('B')
        p_B__A = p_AB.condition('A')
        # products
        p_A__p_B__A_v1 = p_B__A * p_A
        p_A__p_B__A_v2 = p_A * p_B__A
        p_B__p_A__B_v1 = p_B * p_A__B
        p_B__p_A__B_v2 = p_A__B * p_B
        self.assertTrue(series_are_equivalent(p_AB.data, p_A__p_B__A_v1.data))
        self.assertTrue(series_are_equivalent(p_AB.data, p_A__p_B__A_v2.data))
        self.assertTrue(series_are_equivalent(p_AB.data, p_B__p_A__B_v1.data))
        self.assertTrue(series_are_equivalent(p_AB.data, p_B__p_A__B_v2.data))
