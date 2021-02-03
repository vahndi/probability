from unittest.case import TestCase

from probability.distributions import InverseGamma


class TestInverseGamma(TestCase):

    def setUp(self) -> None:

        pass

    def test_fit(self):

        for alpha, beta in zip(
                (1, 2, 3, 3),
                (1, 1, 1, 0.5)
        ):
            inverse_gamma_orig = InverseGamma(alpha=alpha, beta=beta)
            inverse_gamma_fit = InverseGamma.fit(
                inverse_gamma_orig.rvs(100_000)
            )
            self.assertAlmostEqual(inverse_gamma_orig.alpha,
                                   inverse_gamma_fit.alpha, 1)
            self.assertAlmostEqual(inverse_gamma_orig.beta,
                                   inverse_gamma_fit.beta, 1)
