from unittest.case import TestCase

from probability.distributions import Beta, Dirichlet


class BaseTest(TestCase):

    def setUp(self) -> None:

        self.b1 = Beta(700, 300)
        self.b2 = Beta(600, 400)
        self.b3 = Beta(500, 500)
        self.d1 = Dirichlet([500, 300, 200])
        self.d2 = Dirichlet({'x': 100, 'y': 200, 'z': 300})
        self.b1__mul__b2 = self.b1 * self.b2
        self.b3__mul__b1__mul__b2 = self.b3 * self.b1__mul__b2
        self.b1__mul__comp__b1 = self.b1 * (1 - self.b1)
