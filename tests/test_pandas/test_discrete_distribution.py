from itertools import permutations
from unittest import TestCase

from probability.pandas.discrete_distribution import DiscreteDistribution
from tests.shared import read_distribution_data, series_are_equivalent


class TestDiscreteDistribution(TestCase):

    def setUp(self) -> None:

        self.p_abcd = DiscreteDistribution(read_distribution_data('P(A,B,C,D)'))
        self.p_abc = DiscreteDistribution(read_distribution_data('P(A,B,C)'))
        self.p_ab = DiscreteDistribution(read_distribution_data('P(A,B)'))
        self.p_a = DiscreteDistribution(read_distribution_data('P(A)'))
        self.p_d = DiscreteDistribution(read_distribution_data('P(D)'))
        self.p_cd = DiscreteDistribution(read_distribution_data('P(C,D)'))
        self.p_bcd = DiscreteDistribution(read_distribution_data('P(B,C,D)'))
        self.p_abc__d = DiscreteDistribution(read_distribution_data('p(A,B,C|D)'))
        self.p_ab__c__d = DiscreteDistribution(read_distribution_data('p(A,B|C,D)'))
        self.p_a__b__c__d = DiscreteDistribution(read_distribution_data('P(A|B,C,D)'))

    def test_margin(self):

        for n_margins in (1, 2, 3):
            for margin_vars in permutations('ABCD', n_margins):
                p_v = self.p_abcd.margin(*margin_vars)
                self.assertEqual(p_v.name, f'P({",".join(margin_vars)})')
                self.assertEqual(p_v.given_conditions, {})

    def test_given(self):

        p1 = self.p_abcd.given(A=1, B=2)
        p2 = self.p_abcd.given(A=1).given(B=2)
        self.assertTrue(series_are_equivalent(p1.data, p2.data))
        self.assertEqual(p1.name, p2.name)

    def test_condition(self):

        c1 = self.p_abcd.condition('A', 'B')
        c2 = self.p_abcd.condition('B', 'A')
        self.assertTrue(series_are_equivalent(c1.data, c2.data))

    def test_p(self):

        data = self.p_abcd.data
        for _, row in data.reset_index().iterrows():
            row_dict = row.to_dict()
            p = row_dict.pop('p')
            self.assertEqual(p, self.p_abcd.p(**row_dict))

    def test_p_missing_values(self):

        p = self.p_abcd.p(A=1, B=2, C=3, D=4)
        self.assertEqual(0.0, p)

    def test_product_rule(self):

        p_ab = DiscreteDistribution(read_distribution_data('P(A,B)'))
        # margins
        p_a = p_ab.margin('A')
        p_b = p_ab.margin('B')
        # conditions
        p_a__b = p_ab.condition('B')
        p_b__a = p_ab.condition('A')
        # products
        p_a__p_b__a_v1 = p_b__a * p_a
        p_a__p_b__a_v2 = p_a * p_b__a
        p_b__p_a__b_v1 = p_b * p_a__b
        p_b__p_a__b_v2 = p_a__b * p_b
        self.assertTrue(series_are_equivalent(p_ab.data, p_a__p_b__a_v1.data))
        self.assertTrue(series_are_equivalent(p_ab.data, p_a__p_b__a_v2.data))
        self.assertTrue(series_are_equivalent(p_ab.data, p_b__p_a__b_v1.data))
        self.assertTrue(series_are_equivalent(p_ab.data, p_b__p_a__b_v2.data))

    def test_division(self):

        p_abcd_over_d = self.p_abcd / self.p_d
        p_abcd_over_cd = self.p_abcd / self.p_cd
        p_abcd_over_bcd = self.p_abcd / self.p_bcd
        self.assertTrue(series_are_equivalent(self.p_abc__d.data, p_abcd_over_d.data))
        self.assertTrue(series_are_equivalent(self.p_ab__c__d.data, p_abcd_over_cd.data))
        self.assertTrue(series_are_equivalent(self.p_a__b__c__d.data, p_abcd_over_bcd.data))
