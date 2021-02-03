from unittest.case import TestCase

from probability.distributions import Lomax


class TestLomax(TestCase):

    def setUp(self) -> None:

        pass

    def test_fit(self):

        for lambda_, alpha in zip(
                (1, 2, 4, 6),
                (2, 2, 1, 1)
        ):
            lomax_orig = Lomax(lambda_=lambda_, alpha=alpha)
            lomax_fit = Lomax.fit(data=lomax_orig.rvs(100_000))
            self.assertAlmostEqual(lomax_orig.lambda_,
                                   lomax_fit.lambda_, delta=0.2)
            self.assertAlmostEqual(lomax_orig.alpha,
                                   lomax_fit.alpha, delta=0.2)
