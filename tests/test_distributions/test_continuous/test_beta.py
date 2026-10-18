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

    def test_inequalities(self):

        b1 = Beta(2, 2)
        b2 = Beta(5, 2)
        self.assertAlmostEqual(0.5, b1 < 0.5, 5)
        self.assertAlmostEqual(0.5, b1 > 0.5, 5)
        self.assertAlmostEqual(1.0, b1 < 1, 5)
        self.assertAlmostEqual(0.0, b1 > 1, 5)
        self.assertAlmostEqual(1.0, (b1 < b2) + (b1 > b2), 5)

