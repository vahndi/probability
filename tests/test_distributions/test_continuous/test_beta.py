from unittest.case import TestCase

from probability.distributions import Beta


class TestBeta(TestCase):

    def setUp(self) -> None:

        pass

    def test_fit(self):

        for alpha, beta in zip(
            (0.5, 5, 1, 2, 2),
            (0.5, 1, 3, 2, 5)
        ):
            beta_orig = Beta(alpha, beta)
            beta_fit = Beta.fit(beta_orig.rvs(100_000))
            self.assertAlmostEqual(beta_fit.alpha, beta_orig.alpha, 1)
            self.assertAlmostEqual(beta_fit.beta, beta_orig.beta, 1)
