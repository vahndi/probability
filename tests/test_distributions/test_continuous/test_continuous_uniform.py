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
