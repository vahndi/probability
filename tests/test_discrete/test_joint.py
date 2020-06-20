from unittest.case import TestCase

from pandas import DataFrame

from probability.discrete.joint import Joint
from tests.shared import read_distribution


class TestJoint(TestCase):
    
    def setUp(self):
        
        self.p_abcd = Joint.from_series(read_distribution('P(A,B,C,D)'))
        self.p_abc = Joint.from_series(read_distribution('P(A,B,C)'))
        self.p_ab = Joint.from_series(read_distribution('P(A,B)'))
        self.p_a = Joint.from_series(read_distribution('P(A)'))
        self.p_d = Joint.from_series(read_distribution('P(D)'))
        self.p_cd = Joint.from_series(read_distribution('P(C,D)'))
        self.p_bcd = Joint.from_series(read_distribution('P(B,C,D)'))

        # PRML 1.2
        fruit_box_data = DataFrame({
            'box': ['red'] * 8 + ['blue'] * 4,
            'fruit': ['apple'] * 2 + ['orange'] * 7 + ['apple'] * 3
        })
        self.p_bf__obs = Joint.from_observations(fruit_box_data)
        self.p_b__dict = Joint.from_dict(
            data={'blue': 0.6, 'red': 0.4}, variables=['box']
        )
        self.p_bf__dict = Joint.from_dict({
            ('blue', 'apple'): 1 / 4,
            ('blue', 'orange'): 1 / 6,
            ('red', 'apple'): 1 / 12,
            ('red', 'orange'): 1 / 2
        }, variables=['box', 'fruit'])

    def test_init(self):
        
        self.assertEqual(3 ** 4, len(self.p_abcd))
        self.assertEqual(3 ** 3, len(self.p_abc))
        self.assertEqual(3 ** 2, len(self.p_ab))
        self.assertEqual(3, len(self.p_a))
        self.assertEqual(3, len(self.p_d))
        self.assertEqual(3 ** 2, len(self.p_cd))
        self.assertEqual(3 ** 3, len(self.p_bcd))

    def test_variables(self):

        self.assertEqual(['box', 'fruit'], self.p_bf__obs.variables)
        self.assertEqual(['box'], self.p_b__dict.variables)
        self.assertEqual(['box', 'fruit'], self.p_bf__dict.variables)
