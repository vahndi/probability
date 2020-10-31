from unittest.case import TestCase

from pandas import Series

from probability.distributions import Beta
from probability.distributions.conjugate.beta_geometric_conjugate import \
    BetaGeometricConjugate


class TestBetaGeometricConjugate(TestCase):

    def setUp(self) -> None:

        self.series = Series([
            0, 0, 0, 1, 0, 0, 0, 0, 0, 1
        ])

    def test_infer_posterior(self):

        expected = Beta(alpha=1 + 2, beta=1 + 10 - 2)
        actual = BetaGeometricConjugate.infer_posterior(self.series)
        self.assertEqual(expected, actual)
