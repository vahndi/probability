from unittest.case import TestCase

from pandas import Series

from probability.utils import is_binomial


class TestUtils(TestCase):

    def setUp(self) -> None:

        self.int_binomial = Series(data=[0] * 4 + [1] * 6)
        self.float_binomial = self.int_binomial.astype(float)
        self.bool_binomial = self.int_binomial.astype(bool)
        
        self.int_binomial_zeros = Series(data=[0] * 10)
        self.float_binomial_zeros = self.int_binomial_zeros.astype(float)
        self.bool_binomial_zeros = self.int_binomial_zeros.astype(bool)
        
        self.int_binomial_ones = Series(data=[1] * 10)
        self.float_binomial_ones = self.int_binomial_ones.astype(float)
        self.bool_binomial_ones = self.int_binomial_ones.astype(bool)

    def test_is_binomial(self):

        self.assertTrue(is_binomial(self.int_binomial))
        self.assertTrue(is_binomial(self.float_binomial))
        self.assertTrue(is_binomial(self.bool_binomial))

    def test_is_binomial_zeros(self):
        
        self.assertTrue(is_binomial(self.int_binomial_zeros))
        self.assertTrue(is_binomial(self.float_binomial_zeros))
        self.assertTrue(is_binomial(self.bool_binomial_zeros))

    def test_is_binomial_ones(self):
        
        self.assertTrue(is_binomial(self.int_binomial_ones))
        self.assertTrue(is_binomial(self.float_binomial_ones))
        self.assertTrue(is_binomial(self.bool_binomial_ones))
