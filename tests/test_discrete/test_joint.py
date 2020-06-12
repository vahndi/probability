from unittest.case import TestCase

from probability.discrete.joint import Joint
from tests.shared import read_distribution


class TestJoint(TestCase):
    
    def setUp(self):
        
        self.p_abcd = Joint(read_distribution('P(A,B,C,D)'))
        self.p_abc = Joint(read_distribution('P(A,B,C)'))
        self.p_ab = Joint(read_distribution('P(A,B)'))
        self.p_a = Joint(read_distribution('P(A)'))
        self.p_d = Joint(read_distribution('P(D)'))
        self.p_cd = Joint(read_distribution('P(C,D)'))
        self.p_bcd = Joint(read_distribution('P(B,C,D)'))

    def test_init(self):
        
        self.assertEqual(3 ** 4, self.p_abcd.data.shape[0])
        self.assertEqual(3 ** 3, self.p_abc.data.shape[0])
        self.assertEqual(3 ** 2, self.p_ab.data.shape[0])
        self.assertEqual(3, self.p_a.data.shape[0])
        self.assertEqual(3, self.p_d.data.shape[0])
        self.assertEqual(3 ** 2, self.p_cd.data.shape[0])
        self.assertEqual(3 ** 3, self.p_bcd.data.shape[0])
