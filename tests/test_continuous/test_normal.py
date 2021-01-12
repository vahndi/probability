from unittest.case import TestCase


from probability.distributions import Normal


class TestNormal(TestCase):

    def setUp(self) -> None:

        self.n1 = Normal(0, 1)
        self.n2 = Normal(1, 2)

    def test_normal__lt__normal(self):

        self.assertGreater(self.n1 < self.n2, 0.5)
        self.assertLess(self.n2 < self.n1, 0.5)

    def test_normal__gt__normal(self):

        self.assertLess(self.n1 > self.n2, 0.5)
        self.assertGreater(self.n2 > self.n1, 0.5)

    def test_normal__lt__float(self):

        self.assertEqual(self.n1 < 0, 0.5)
        self.assertEqual(self.n2 < 1, 0.5)

    def test_normal__gt__float(self):

        self.assertEqual(self.n1 > 0, 0.5)
        self.assertEqual(self.n2 > 1, 0.5)

    def test_normal__eq__normal(self):

        self.assertTrue(self.n1 == self.n1)
        self.assertFalse(self.n1 == self.n2)

    def test_normal__ne__normal(self):

        self.assertFalse(self.n1 != self.n1)
        self.assertTrue(self.n1 != self.n2)

    def test_fit(self):

        for mu, sigma_sq in zip(
                (0, 0, 0, -2),
                (0.2, 1.0, 5.0, 0.5)
        ):
            normal_orig = Normal(mu=mu, sigma_sq=sigma_sq)
            normal_fit = Normal.fit(normal_orig.rvs(100_000))
            self.assertAlmostEqual(normal_orig.mu, normal_fit.mu, 1)
            self.assertAlmostEqual(normal_orig.sigma_sq, normal_fit.sigma_sq, 1)
