from itertools import product, permutations

from pandas import ExcelFile, read_excel, Series
import unittest
from unittest import TestCase

from probability.pandas.prob_utils import margin, condition
from tests.paths import FN_PANDAS_TESTS


class TestProbUtils(TestCase):

    def setUp(self) -> None:

        self.xlsx = ExcelFile(str(FN_PANDAS_TESTS))
        self.vars = ['A', 'B', 'C', 'D']
        self.joint = self._read_distribution('P(A,B,C,D)')
        self.p_A = self._read_distribution('P(A)')
        self.p_AB = self._read_distribution('P(A,B)')
        self.p_ABC = self._read_distribution('P(A,B,C)')
        self.p_ABC__D_1 = self._read_distribution('p(A,B,C|D=1)')
        self.p_AB__C_1__D_2 = self._read_distribution('p(A,B|C=1,D=2)')
        self.p_A__B_1__C_2__D_3 = self._read_distribution('p(A|B=1,C=2,D=3)')
        self.p_ABC__D = self._read_distribution('p(A,B,C|D)')
        self.p_AB__C__D = self._read_distribution('p(A,B|C,D)')
        self.p_A__B__C__D = self._read_distribution('P(A|B,C,D)')

    def _read_distribution(self, sheet_name: str) -> Series:

        data = read_excel(self.xlsx, sheet_name)
        vars = [c for c in data.columns if c != 'p' and not c.startswith('_')]
        return data.set_index(vars)['p']

    def _series_are_equivalent(self, series_1: Series, series_2: Series):
        """
        Determine if the 2 series share the same items.
        N.B. where series 1 has p(0), the associated item may be missing from series 2.
        """
        d1 = series_1.copy().reset_index()
        cols_1 = sorted([c for c in d1.columns if c != 'p'])
        cols_p = cols_1 + ['p']
        s1 = d1[cols_p].set_index(cols_1)['p']
        d2 = series_2.copy().reset_index()
        cols_2 = sorted([c for c in d2.columns if c != 'p'])
        if cols_1 != cols_2:
            return False
        s2 = d2[cols_p].set_index(cols_2)['p']
        for k, v in s1.iteritems():
            if v == 0:
                continue
            if k not in s2.keys() or s2[k] != v:
                return False
        return True

    def test_margins(self):

        self.assertTrue(self._series_are_equivalent(self.p_A, margin(self.joint, 'A')))
        for m in ['B', 'C', 'D']:
            self.assertFalse(self._series_are_equivalent(self.p_A, margin(self.joint, m)))

        self.assertTrue(self._series_are_equivalent(self.p_AB, margin(self.joint, 'A', 'B')))
        self.assertTrue(self._series_are_equivalent(self.p_AB, margin(self.joint, 'B', 'A')))
        for m1, m2 in product(self.vars, self.vars):
            if m1 == m2 or {m1, m2} == {'A', 'B'}:
                continue
            self.assertFalse(self._series_are_equivalent(self.p_AB, margin(self.joint, m1, m2)))

        for m1, m2, m3 in permutations(['A', 'B', 'C']):
            self.assertTrue(self._series_are_equivalent(self.p_ABC, margin(self.joint, m1, m2, m3)))
        for m1, m2, m3 in product(self.vars, self.vars, self.vars):
            if len({m1, m2, m3}) != 3 or {m1, m2, m3} == {'A', 'B', 'C'}:
                continue
            self.assertFalse(self._series_are_equivalent(self.p_ABC, margin(self.joint, m1, m2, m3)))

    def test_given_conditions(self):

        self.assertTrue(self._series_are_equivalent(self.p_ABC__D_1, condition(self.joint, D=1)))
        for c in ['A', 'B', 'C']:
            kwargs = {c: 1}
            self.assertFalse(self._series_are_equivalent(self.p_ABC__D_1, condition(self.joint, **kwargs)))

        self.assertTrue(self._series_are_equivalent(self.p_AB__C_1__D_2, condition(self.joint, C=1, D=2)))
        for c1, c2 in product(self.vars, self.vars):
            if c1 == c2 or (c1 == 'C' and c2 == 'D'):
                continue
            kwargs = {c1: 1, c2: 2}
            self.assertFalse(self._series_are_equivalent(self.p_AB__C_1__D_2, condition(self.joint, **kwargs)))

        self.assertTrue(self._series_are_equivalent(self.p_A__B_1__C_2__D_3, condition(self.joint, B=1, C=2, D=3)))
        for c1, c2, c3 in product(self.vars, self.vars, self.vars):
            if len({c1, c2, c3}) != 3 or (c1 == 'B' and c2 == 'C' and c3 == 'D'):
                continue
            kwargs = {c1: 1, c2: 2, c3: 3}
            self.assertFalse(self._series_are_equivalent(self.p_A__B_1__C_2__D_3, condition(self.joint, **kwargs)))

    def test_not_given_conditions(self):

        self.assertTrue(self._series_are_equivalent(self.p_ABC__D, condition(self.joint, 'D')))
        for c in ['A', 'B', 'C']:
            self.assertFalse(self._series_are_equivalent(self.p_ABC__D, condition(self.joint, c)))

        self.assertTrue(self._series_are_equivalent(self.p_AB__C__D, condition(self.joint, 'C', 'D')))
        self.assertTrue(self._series_are_equivalent(self.p_AB__C__D, condition(self.joint, 'D', 'C')))
        for c1, c2 in product(self.vars, self.vars):
            if c1 == c2 or {c1, c2} == {'C', 'D'}:
                continue
            self.assertFalse(self._series_are_equivalent(self.p_AB__C__D, condition(self.joint, c1, c2)))

        for c1, c2, c3 in permutations(['B', 'C', 'D']):
            self.assertTrue(self._series_are_equivalent(self.p_A__B__C__D, condition(self.joint, c1, c2, c3)))
        for c1, c2, c3 in product(self.vars, self.vars, self.vars):
            if len({c1, c2, c3}) != 3 or {c1, c2, c3} == {'B', 'C', 'D'}:
                continue
            self.assertFalse(self._series_are_equivalent(self.p_A__B__C__D, condition(self.joint, c1, c2, c3)))

    def test_mixed_conditions(self):

        pass


if __name__ == '__main__':

    unittest.main()
