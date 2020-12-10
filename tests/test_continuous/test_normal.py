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
