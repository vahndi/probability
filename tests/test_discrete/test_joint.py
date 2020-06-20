from unittest.case import TestCase

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

    def test_init(self):
        
        self.assertEqual(3 ** 4, len(self.p_abcd))
        self.assertEqual(3 ** 3, len(self.p_abc))
        self.assertEqual(3 ** 2, len(self.p_ab))
        self.assertEqual(3, len(self.p_a))
        self.assertEqual(3, len(self.p_d))
        self.assertEqual(3 ** 2, len(self.p_cd))
        self.assertEqual(3 ** 3, len(self.p_bcd))
