from unittest.case import TestCase

from pandas import Series

from probability.distributions import Multinomial


class TestMultinomial(TestCase):

    def setUp(self) -> None:

        self.p = Series({'a': 0.4, 'b': 0.3, 'c': 0.2, 'd': 0.1})
        self.m_array = Multinomial(n=10, p=self.p.values)
        self.m_series = Multinomial(n=10, p=self.p)

    def test_init_with_array(self):

        expected = Series({'x1': 0.4, 'x2': 0.3, 'x3': 0.2, 'x4': 0.1})
        actual = self.m_array.p
        self.assertTrue(expected.equals(actual))

    def test_init_with_series(self):

        expected = self.p
        actual = self.m_series.p
        self.assertTrue(expected.equals(actual))

    def test_set_alpha_with_array(self):

        m = Multinomial(n=10, p=[0.1, 0.2, 0.3, 0.4])
        expected = Series({'x1': 0.4, 'x2': 0.3, 'x3': 0.2, 'x4': 0.1})
        m.p = [0.4, 0.3, 0.2, 0.1]
        actual = m.p
        self.assertTrue(expected.equals(actual))

    def test_set_alpha_with_series(self):

        m = Multinomial(n=10, p=[0.1, 0.2, 0.3, 0.4])
        expected = Series({'x1': 0.4, 'x2': 0.3, 'x3': 0.2, 'x4': 0.1})
        m.p = expected
        actual = m.p
        self.assertTrue(expected.equals(actual))

    def test_str(self):

        self.assertEqual(
            'Multinomial(x1=0.4, x2=0.3, x3=0.2, x4=0.1)',
            str(self.m_array)
        )
        self.assertEqual(
            'Multinomial(a=0.4, b=0.3, c=0.2, d=0.1)',
            str(self.m_series)
        )