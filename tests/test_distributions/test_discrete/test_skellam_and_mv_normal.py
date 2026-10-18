from unittest.case import TestCase

from probability.distributions import Skellam, Poisson, NegativeBinomial
from probability.distributions.multivariate.mv_normal import MVNormal


class TestSkellamAndMVNormal(TestCase):

    def test_skellam_properties_and_str(self):

        sk = Skellam(mu_1=4.0, mu_2=2.0)
        self.assertEqual(4.0, sk.mu_1)
        self.assertEqual(2.0, sk.mu_2)
        self.assertEqual(2.0, sk.mean())
        self.assertEqual('Skellam(μ1=4, μ2=2)', str(sk))
        sk.mu_1 = 6.0
        self.assertEqual(4.0, sk.mean())
        sk.mu_2 = 1.0
        self.assertEqual(5.0, sk.mean())

    def test_poisson_and_neg_binom_lower_bounds(self):

        self.assertEqual(0, Poisson(3.0).lower_bound)
        self.assertEqual(0, NegativeBinomial(3, 0.5).lower_bound)

    def test_mv_normal_setters_update_distribution(self):

        mvn = MVNormal(mu=[0.0, 0.0], sigma=[[1.0, 0.0], [0.0, 1.0]])
        self.assertAlmostEqual(0.0, mvn.pdf().at([[10.0, 10.0]]).iloc[0], 5)
        mvn.mu = [10.0, 10.0]
        self.assertGreater(mvn.pdf().at([[10.0, 10.0]]).iloc[0], 0.1)
