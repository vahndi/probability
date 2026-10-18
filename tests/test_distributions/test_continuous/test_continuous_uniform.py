from unittest.case import TestCase

from probability.distributions import ContinuousUniform


class TestContinuousUniform(TestCase):

    def setUp(self) -> None:

        pass

    def test_fit(self):

        uniform = ContinuousUniform(2, 5)
        uniform_fit = ContinuousUniform.fit(uniform.rvs(100_000))
        self.assertAlmostEqual(uniform.a, uniform_fit.a, 1)
        self.assertAlmostEqual(uniform.b, uniform_fit.b, 1)

    def test_fit_fixed_params(self):

        uniform = ContinuousUniform(2, 5)
        samples = uniform.rvs(10_000, random_state=42)
        fit_a = ContinuousUniform.fit(samples, a=2)
        self.assertEqual(2, fit_a.a)
        self.assertAlmostEqual(5, fit_a.b, 1)
        fit_ab = ContinuousUniform.fit(samples, a=2, b=5)
        self.assertEqual(2, fit_ab.a)
        self.assertEqual(5, fit_ab.b)

