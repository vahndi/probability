from unittest.case import TestCase

from probability.distributions import BetaBinomial


class TestBetaBinomial(TestCase):

    def setUp(self) -> None:

        pass

    def test_fits(self):

        for alpha, beta in zip(
                (0.2, 0.7, 2),
                (0.25, 2, 2)
        ):
            bb_orig = BetaBinomial(n=10, alpha=alpha, beta=beta)
            bb_fits = BetaBinomial.fits(bb_orig.rvs(100_000), n=10)
            self.assertAlmostEqual(bb_orig.alpha, bb_fits.alpha, 1)
            self.assertAlmostEqual(bb_orig.beta, bb_fits.beta, 1)
