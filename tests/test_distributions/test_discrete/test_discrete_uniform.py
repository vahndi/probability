from unittest.case import TestCase

from probability.distributions import DiscreteUniform


class TestDiscreteUniform(TestCase):

    def setUp(self) -> None:
        self.d1 = DiscreteUniform(a=1, b=6)  # Standard 6-sided die
        self.d2 = DiscreteUniform(a=0, b=10)

    def test_properties(self):
        self.assertEqual(self.d1.a, 1)
        self.assertEqual(self.d1.b, 6)
        self.assertEqual(self.d1.n, 6)
        self.assertEqual(self.d1.lower_bound, 1)
        self.assertEqual(self.d1.upper_bound, 6)
        self.assertEqual(self.d1.mean(), 3.5)

    def test_pmf_cdf(self):
        # 1/6 for each face
        for k in range(1, 7):
            self.assertAlmostEqual(self.d1.pmf().at(k), 1.0 / 6.0, places=5)
        self.assertAlmostEqual(self.d1.cdf().at(3), 0.5, places=5)

    def test_fit(self):
        rolls = [1, 2, 3, 4, 5, 6]
        fitted = DiscreteUniform.fit(rolls)
        self.assertEqual(fitted.a, 1)
        self.assertEqual(fitted.b, 6)

    def test_equality(self):
        d_same = DiscreteUniform(a=1, b=6)
        self.assertEqual(self.d1, d_same)
        self.assertNotEqual(self.d1, self.d2)

    def test_invalid_bounds(self):
        with self.assertRaises(ValueError):
            DiscreteUniform(a=10, b=5)
