from unittest import TestCase

from probability.pandas import DiscreteDistribution
from tests.shared import read_distribution_data, series_are_equivalent


class TestConditionalTable(TestCase):

    def setUp(self) -> None:

        self.p_abcd = DiscreteDistribution(read_distribution_data('P(A,B,C,D)'))

    def test_margin(self):

        mc = self.p_abcd.margin('A', 'B', 'C').condition('C')
        cm = self.p_abcd.condition('C').margin('A', 'B')
        self.assertTrue(series_are_equivalent(mc.data, cm.data))

    def test_given(self):

        g = self.p_abcd.given(C=1, D=2)
        cg = self.p_abcd.condition('C', 'D').given(C=1, D=2)
        self.assertTrue(series_are_equivalent(g.data, cg.data))

    def test_p(self):

        p = self.p_abcd.given(C=3, D=2).p(A=1, B=2)
        cp = self.p_abcd.condition('C', 'D').p(A=1, B=2, C=3, D=2)
        self.assertAlmostEqual(p, cp)
