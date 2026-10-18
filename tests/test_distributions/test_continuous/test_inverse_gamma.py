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

    def test_fit_fixed_params(self):

        ig = InverseGamma(alpha=3.0, beta=2.0)
        samples = ig.rvs(50_000, random_state=42)
        ig_fixed_beta = InverseGamma.fit(samples, beta=2.0)
        self.assertEqual(2.0, ig_fixed_beta.beta)
        self.assertAlmostEqual(3.0, ig_fixed_beta.alpha, 1)

    def test_entropy_and_log_sf(self):

        ig = InverseGamma(alpha=3.0, beta=2.0)
        self.assertIsInstance(ig.entropy(), float)
        import numpy as np
        self.assertAlmostEqual(
            np.log(ig.sf().at(1.5)),
            ig.log_sf().at(1.5),
            6
        )

