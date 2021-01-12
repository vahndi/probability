from unittest.case import TestCase

from probability.distributions import Gamma


class TestGamma(TestCase):

    def setUp(self) -> None:

        pass

    def test_fit(self):

        for k, theta in zip(
                (1, 2, 3, 5, 9, 7.5, 0.5),
                (2, 2, 2, 1, 0.5, 1, 1)
        ):
            gamma_orig = Gamma.from_k_theta(k=k, theta=theta)
            gamma_fit = Gamma.fit(gamma_orig.rvs(100_000))
            self.assertAlmostEqual(gamma_orig.alpha, gamma_fit.alpha, delta=0.1)
            self.assertAlmostEqual(gamma_orig.beta, gamma_fit.beta, delta=0.1)
