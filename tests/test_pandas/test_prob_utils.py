from itertools import product, permutations
from pandas import ExcelFile
import unittest
from unittest import TestCase

from probability.pandas.prob_utils import margin, condition, multiply, given

from tests.paths import FN_PANDAS_TESTS
from tests.shared import read_distribution_data, series_are_equivalent


class TestProbUtils(TestCase):

    def setUp(self) -> None:

        self.xlsx = ExcelFile(str(FN_PANDAS_TESTS))
        self.vars = ['A', 'B', 'C', 'D']
        # joints
        self.joint = read_distribution_data('P(A,B,C,D)')
        # marginals
        self.p_A = read_distribution_data('P(A)')
        self.p_AB = read_distribution_data('P(A,B)')
        self.p_ABC = read_distribution_data('P(A,B,C)')
        # conditionals
        self.p_ABC__D_1 = read_distribution_data('p(A,B,C|D=1)')
        self.p_AB__C_1__D_2 = read_distribution_data('p(A,B|C=1,D=2)')
        self.p_A__B_1__C_2__D_3 = read_distribution_data('p(A|B=1,C=2,D=3)')
        self.p_ABC__D = read_distribution_data('p(A,B,C|D)')
        self.p_AB__C__D = read_distribution_data('p(A,B|C,D)')
        self.p_A__B__C__D = read_distribution_data('P(A|B,C,D)')
        self.p_AB__C__D_1 = read_distribution_data('p(A,B|C,D=1)')
        self.p_AB__C_2__D = read_distribution_data('p(A,B|C=2,D)')

    def test_margins(self):

        self.assertTrue(series_are_equivalent(self.p_A, margin(self.joint, 'A')))
        for m in ['B', 'C', 'D']:
            self.assertFalse(series_are_equivalent(self.p_A, margin(self.joint, m)))

        self.assertTrue(series_are_equivalent(self.p_AB, margin(self.joint, 'A', 'B')))
        self.assertTrue(series_are_equivalent(self.p_AB, margin(self.joint, 'B', 'A')))
        for m1, m2 in product(self.vars, self.vars):
            if m1 == m2 or {m1, m2} == {'A', 'B'}:
                continue
            self.assertFalse(series_are_equivalent(self.p_AB, margin(self.joint, m1, m2)))

        for m1, m2, m3 in permutations(['A', 'B', 'C']):
            self.assertTrue(series_are_equivalent(self.p_ABC, margin(self.joint, m1, m2, m3)))
        for m1, m2, m3 in product(self.vars, self.vars, self.vars):
            if len({m1, m2, m3}) != 3 or {m1, m2, m3} == {'A', 'B', 'C'}:
                continue
            self.assertFalse(series_are_equivalent(self.p_ABC, margin(self.joint, m1, m2, m3)))

    def test_given_conditions(self):

        self.assertTrue(series_are_equivalent(self.p_ABC__D_1, given(self.joint, D=1)))
        for c in ['A', 'B', 'C']:
            kwargs = {c: 1}
            self.assertFalse(series_are_equivalent(self.p_ABC__D_1, given(self.joint, **kwargs)))

        self.assertTrue(series_are_equivalent(self.p_AB__C_1__D_2, given(self.joint, C=1, D=2)))
        for c1, c2 in product(self.vars, self.vars):
            if c1 == c2 or (c1 == 'C' and c2 == 'D'):
                continue
            kwargs = {c1: 1, c2: 2}
            self.assertFalse(series_are_equivalent(self.p_AB__C_1__D_2, given(self.joint, **kwargs)))

        self.assertTrue(series_are_equivalent(self.p_A__B_1__C_2__D_3, given(self.joint, B=1, C=2, D=3)))

    def test_not_given_conditions(self):

        self.assertTrue(series_are_equivalent(self.p_ABC__D, condition(self.joint, 'D')))
        for c in ['A', 'B', 'C']:
            self.assertFalse(series_are_equivalent(self.p_ABC__D, condition(self.joint, c)))

        self.assertTrue(series_are_equivalent(self.p_AB__C__D, condition(self.joint, 'C', 'D')))
        self.assertTrue(series_are_equivalent(self.p_AB__C__D, condition(self.joint, 'D', 'C')))
        for c1, c2 in product(self.vars, self.vars):
            if c1 == c2 or {c1, c2} == {'C', 'D'}:
                continue
            self.assertFalse(series_are_equivalent(self.p_AB__C__D, condition(self.joint, c1, c2)))

        for c1, c2, c3 in permutations(['B', 'C', 'D']):
            self.assertTrue(series_are_equivalent(self.p_A__B__C__D, condition(self.joint, c1, c2, c3)))
        for c1, c2, c3 in product(self.vars, self.vars, self.vars):
            if len({c1, c2, c3}) != 3 or {c1, c2, c3} == {'B', 'C', 'D'}:
                continue
            self.assertFalse(series_are_equivalent(self.p_A__B__C__D, condition(self.joint, c1, c2, c3)))

    def test_mixed_conditions(self):

        self.assertTrue(series_are_equivalent(self.p_AB__C__D_1, condition(given(self.joint, D=1), 'C')))
        self.assertTrue(series_are_equivalent(self.p_AB__C_2__D, condition(given(self.joint, C=2), 'D')))

    def test_chain_not_given_conditions(self):

        self.assertTrue(series_are_equivalent(
            condition(self.joint, 'C', 'D'),
            condition(condition(self.joint, 'C'), 'C', 'D')  # need to recondition on any existing conditions
        ))

    def test_chain_given_conditions(self):

        self.assertTrue(series_are_equivalent(
            given(given(self.joint, C=1), D=2),
            given(given(self.joint, D=2), C=1)
        ))

    def test_chain_mixed_conditions(self):

        self.assertFalse(series_are_equivalent(
            condition(given(self.joint, C=1), 'D'),
            given(condition(self.joint, 'D'), C=1)
        ))

    def test_multiply(self):

        p_AB = read_distribution_data('P(A,B)')
        p_A = margin(p_AB, 'A')
        p_B = margin(p_AB, 'B')
        p_A__B = condition(p_AB, 'B')
        p_B__A = condition(p_AB, 'A')
        p_A__B_p_B = multiply(p_A__B, p_B)
        p_B__A_p_A = multiply(p_B__A, p_A)
        self.assertTrue(series_are_equivalent(p_A__B_p_B, p_AB))
        self.assertTrue(series_are_equivalent(p_B__A_p_A, p_AB))


if __name__ == '__main__':

    unittest.main()
