from unittest.case import TestCase

from probability.distributions import Geometric


class TestGeometric(TestCase):

    def setUp(self) -> None:

        pass

    def test_fits(self):

        for p in (0.2, 0.5, 0.8):
            geom_orig = Geometric(p=p)
            geom_fits = Geometric.fits(geom_orig.rvs(100_000))
            self.assertAlmostEqual(geom_orig.p, geom_fits.p, 1)
