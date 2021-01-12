from unittest.case import TestCase

from probability.distributions import PERT


class TestPERT(TestCase):

    def setUp(self) -> None:

        pass

    def test_fit(self):

        for a, b, c in zip(
                (0, 0, 0),
                (10, 50, 70),
                (100, 100, 100)
        ):
            pert_orig = PERT(a=a, b=b, c=c)
            pert_fit = PERT.fit(pert_orig.rvs(100_000))
            self.assertAlmostEqual(pert_orig.a / 100, pert_fit.a / 100, 1)
            self.assertAlmostEqual(pert_orig.b / 100, pert_fit.b / 100, 1)
            self.assertAlmostEqual(pert_orig.c / 100, pert_fit.c / 100, 1)
