from unittest.case import TestCase

from probability.distributions import Exponential


class TestExponential(TestCase):

    def setUp(self) -> None:

        pass

    def test_fit(self):

        for lambda_ in (0.5, 1, 1.5):
            exponential = Exponential(lambda_=lambda_)
            exponential_fit = Exponential.fit(exponential.rvs(100_000))
            self.assertAlmostEqual(exponential.lambda_,
                                   exponential_fit.lambda_, 1)
