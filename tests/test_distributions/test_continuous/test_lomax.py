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
            lomax_fit = Lomax.fit(data=lomax_orig.rvs(100_000, random_state=0))
            self.assertAlmostEqual(lomax_orig.lambda_,
                                   lomax_fit.lambda_, delta=0.2)
            self.assertAlmostEqual(lomax_orig.alpha,
                                   lomax_fit.alpha, delta=0.2)

    def test_fit_fixed_params(self):

        lomax_orig = Lomax(lambda_=2.0, alpha=3.0)
        data = lomax_orig.rvs(20_000, random_state=0)
        lomax_fit = Lomax.fit(data=data, alpha=3.0)
        self.assertEqual(3.0, lomax_fit.alpha)
        self.assertAlmostEqual(2.0, lomax_fit.lambda_, delta=0.2)

