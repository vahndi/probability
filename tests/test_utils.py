from unittest.case import TestCase

from pandas import Series

from probability.utils import is_binary


class TestUtils(TestCase):

    def setUp(self) -> None:

        self.int_binary = Series(data=[0] * 4 + [1] * 6)
        self.float_binary = self.int_binary.astype(float)
        self.bool_binary = self.int_binary.astype(bool)
        
        self.int_binary_zeros = Series(data=[0] * 10)
        self.float_binary_zeros = self.int_binary_zeros.astype(float)
        self.bool_binary_zeros = self.int_binary_zeros.astype(bool)
        
        self.int_binary_ones = Series(data=[1] * 10)
        self.float_binary_ones = self.int_binary_ones.astype(float)
        self.bool_binary_ones = self.int_binary_ones.astype(bool)

    def test_is_binary(self):

        self.assertTrue(is_binary(self.int_binary))
        self.assertTrue(is_binary(self.float_binary))
        self.assertTrue(is_binary(self.bool_binary))

    def test_is_binary_zeros(self):
        
        self.assertTrue(is_binary(self.int_binary_zeros))
        self.assertTrue(is_binary(self.float_binary_zeros))
        self.assertTrue(is_binary(self.bool_binary_zeros))

    def test_is_binary_ones(self):
        
        self.assertTrue(is_binary(self.int_binary_ones))
        self.assertTrue(is_binary(self.float_binary_ones))
        self.assertTrue(is_binary(self.bool_binary_ones))
