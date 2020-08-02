from unittest.case import TestCase

from pandas import Series

from probability.distributions import Dirichlet, Beta


class TestDirichlet(TestCase):

    def setUp(self) -> None:
        self.alpha = Series({'a': 0.4, 'b': 0.3, 'c': 0.2, 'd': 0.1})
        self.d_array = Dirichlet(alpha=self.alpha.values)
        self.d_series = Dirichlet(alpha=self.alpha)
        self.d_dict = Dirichlet(alpha=self.alpha.to_dict())

    def test_init_with_array(self):
        expected = Series({'α1': 0.4, 'α2': 0.3, 'α3': 0.2, 'α4': 0.1})
        actual = self.d_array.alpha
        self.assertTrue(expected.equals(actual))

    def test_init_with_series(self):

        expected = self.alpha
        actual = self.d_series.alpha
        self.assertTrue(expected.equals(actual))

    def test_init_with_dict(self):

        expected = self.alpha
        actual = self.d_dict.alpha
        self.assertTrue(expected.equals(actual))

    def test_set_alpha_with_array(self):

        d = Dirichlet([0.1, 0.2, 0.3, 0.4])
        expected = Series({'α1': 0.4, 'α2': 0.3, 'α3': 0.2, 'α4': 0.1})
        d.alpha = [0.4, 0.3, 0.2, 0.1]
        actual = d.alpha
        self.assertTrue(expected.equals(actual))

    def test_set_alpha_with_series(self):
        d = Dirichlet([0.1, 0.2, 0.3, 0.4])
        expected = Series({'α1': 0.4, 'α2': 0.3, 'α3': 0.2, 'α4': 0.1})
        d.alpha = expected
        actual = d.alpha
        self.assertTrue(expected.equals(actual))

    def test_str(self):

        self.assertEqual(
            'Dirichlet(α1=0.4, α2=0.3, α3=0.2, α4=0.1)',
            str(self.d_array)
        )
        self.assertEqual(
            'Dirichlet(a=0.4, b=0.3, c=0.2, d=0.1)',
            str(self.d_series)
        )

    def test_get_item(self):

        for k, v in self.d_series.alpha.items():
            expected = Beta(alpha=v, beta=1-v)
            actual = self.d_series[k]
            self.assertTrue(expected == actual)
