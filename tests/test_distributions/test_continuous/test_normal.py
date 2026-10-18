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
            normal_fit = Normal.fit(normal_orig.rvs(100_000, random_state=42))
            self.assertAlmostEqual(normal_orig.mu, normal_fit.mu, 1)
            self.assertAlmostEqual(normal_orig.sigma_sq, normal_fit.sigma_sq, 1)

    def test_normal_arithmetic(self):
        # Chapter 2 example: N(1, 4^2) - N(4, 3^2) == N(-3, 5^2)
        n1 = Normal(mu=4, sigma=3)
        n2 = Normal(mu=1, sigma=4)
        n_diff = n2 - n1
        self.assertIsInstance(n_diff, Normal)
        self.assertEqual(Normal(mu=-3, sigma=5), n_diff)

        n_sum = n1 + n2
        self.assertEqual(Normal(mu=5, sigma=5), n_sum)

        # Scalar operations and linear combinations
        self.assertEqual(Normal(mu=6, sigma=3), n1 + 2)
        self.assertEqual(Normal(mu=6, sigma=3), 2 + n1)
        self.assertEqual(Normal(mu=2, sigma=3), n1 - 2)
        self.assertEqual(Normal(mu=-2, sigma=3), 2 - n1)
        self.assertEqual(Normal(mu=-4, sigma=3), -n1)
        self.assertEqual(Normal(mu=-8, sigma=6), -2 * n1)
        self.assertEqual(Normal(mu=8, sigma=6), n1 * 2)
        self.assertEqual(Normal(mu=2, sigma=1.5), n1 / 2)

        # Preserves μσ² parameterization when both operands use μσ²
        n1_sq = Normal(mu=4, sigma_sq=9)
        n2_sq = Normal(mu=1, sigma_sq=16)
        diff_sq = n2_sq - n1_sq
        self.assertEqual('Normal(mu=-3, sigma_sq=25.0)', repr(diff_sq))

    def test_normal_plot_ci_and_fill_interval(self):
        import matplotlib.pyplot as plt
        std_norm = Normal.StandardNormal()
        ax1 = std_norm.plot(ci=[0.68, 0.95, 0.997])
        self.assertGreaterEqual(len(ax1.collections), 3)
        plt.close(ax1.figure)

        ax2 = std_norm.plot(fill_interval=(1.96, None))
        self.assertGreaterEqual(len(ax2.collections), 1)
        plt.close(ax2.figure)


