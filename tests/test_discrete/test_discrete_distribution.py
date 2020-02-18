from itertools import permutations
from unittest import TestCase

from probability.discrete.discrete_distribution import DiscreteDistribution
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

    def test_p_2(self):

        self.assertAlmostEqual(0.33, self.p_a.p(A=1))
        self.assertAlmostEqual(0.36, self.p_a.p(A=2))
        self.assertAlmostEqual(0.31, self.p_a.p(A=3))
        self.assertAlmostEqual(0.33, self.p_a.p(A__lt=2))
        self.assertAlmostEqual(0.31, self.p_a.p(A__gt=2))
        self.assertAlmostEqual(0.33 + 0.31, self.p_a.p(A__ne=2))
        self.assertAlmostEqual(0.33 + 0.36, self.p_a.p(A__le=2))
        self.assertAlmostEqual(0.36 + 0.31, self.p_a.p(A__ge=2))
        self.assertAlmostEqual(0.33 + 0.31, self.p_a.p(A__in=[1, 3]))
        self.assertAlmostEqual(0.36, self.p_a.p(A__not_in=[1, 3]))

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

    def test_multiplication(self):

        p_a = DiscreteDistribution.from_dict({'a1': 0.3, 'a2': 0.7}, var_names='a')
        p_b = DiscreteDistribution.from_dict({'b1': 0.4, 'b2': 0.6}, var_names='b')
        p_c = DiscreteDistribution.from_dict({'c1': 0.2, 'c2': 0.8}, var_names='c')
        p_ab = DiscreteDistribution.from_dict({
            ('a1', 'b1'): 0.3 * 0.4,
            ('a1', 'b2'): 0.3 * 0.6,
            ('a2', 'b1'): 0.7 * 0.4,
            ('a2', 'b2'): 0.7 * 0.6,
        }, var_names=['a', 'b'])
        p_abc = DiscreteDistribution.from_dict({
            ('a1', 'b1', 'c1'): 0.3 * 0.4 * 0.2,
            ('a1', 'b1', 'c2'): 0.3 * 0.4 * 0.8,
            ('a1', 'b2', 'c1'): 0.3 * 0.6 * 0.2,
            ('a1', 'b2', 'c2'): 0.3 * 0.6 * 0.8,
            ('a2', 'b1', 'c1'): 0.7 * 0.4 * 0.2,
            ('a2', 'b1', 'c2'): 0.7 * 0.4 * 0.8,
            ('a2', 'b2', 'c1'): 0.7 * 0.6 * 0.2,
            ('a2', 'b2', 'c2'): 0.7 * 0.6 * 0.8
        }, var_names=['a', 'b', 'c'])
        self.assertEqual(p_ab.data.to_dict(), (p_a * p_b).data.to_dict())
        self.assertEqual(p_ab.name, (p_a * p_b).name)
        self.assertEqual((p_ab * p_c).data.to_dict(), p_abc.data.to_dict())
