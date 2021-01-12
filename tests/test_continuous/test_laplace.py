from unittest.case import TestCase

from probability.distributions import Laplace


class TestLaplace(TestCase):

    def setUp(self) -> None:

        pass

    def test_fit(self):

        for mu, b in zip(
                (0, 0, 0, -5),
                (1, 2, 4, 4)
        ):
            laplace_orig = Laplace(mu=mu, b=b)
            laplace_fit = Laplace.fit(laplace_orig.rvs(100_000))
            self.assertAlmostEqual(laplace_orig.mu, laplace_fit.mu, 1)
            self.assertAlmostEqual(laplace_fit.b, laplace_fit.b, 1)
