from unittest.case import TestCase

from probability.distributions import Binomial


class TestBinomial(TestCase):

    def setUp(self) -> None:

        self.b1 = Binomial(10, 0.5)
        self.b2 = Binomial(10, 0.7)

    def test_binomial__lt__binomial(self):

        self.assertGreater(self.b1 < self.b2, 0.5)
        self.assertLess(self.b2 < self.b1, 0.5)

    def test_binomial__gt__binomial(self):

        self.assertLess(self.b1 > self.b2, 0.5)
        self.assertGreater(self.b2 > self.b1, 0.5)

    def test_binomial__lt__int(self):

        self.assertAlmostEqual(self.b1 < 6, 0.623048, 3)
        self.assertAlmostEqual(self.b2 < 6,  0.150269, 3)

    def test_binomial__gt__int(self):

        self.assertAlmostEqual(self.b1 > 5, 1 - 0.623048, 3)
        self.assertAlmostEqual(self.b2 > 5, 1 - 0.150269, 3)

    def test_binomial__le__int(self):

        self.assertAlmostEqual(self.b1 <= 5, 0.623048, 3)
        self.assertAlmostEqual(self.b2 <= 5, 0.150269, 3)

    def test_binomial__ge__int(self):

        self.assertAlmostEqual(self.b1 >= 6, 1 - 0.623048, 3)
        self.assertAlmostEqual(self.b2 >= 6, 1 - 0.150269, 3)

    def test_binomial__eq__binomial(self):

        self.assertTrue(self.b1 == self.b1)
        self.assertFalse(self.b1 == self.b2)

    def test_binomial__ne__binomial(self):

        self.assertFalse(self.b1 != self.b1)
        self.assertTrue(self.b1 != self.b2)

    def test_fits(self):

        for n, p in zip(
                (20, 20, 40),
                (0.5, 0.7, 0.5)
        ):
            binom_orig = Binomial(n=n, p=p)
            binom_fit = Binomial.fits(binom_orig.rvs(100_000), n=n)
            self.assertAlmostEqual(binom_orig.p, binom_fit.p, 1)
