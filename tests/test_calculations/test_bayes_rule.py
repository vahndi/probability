from unittest.case import TestCase

from pandas import Series

from probability.calculations.bayes_rule import BayesRule


class TestBayesRule(TestCase):

    def setUp(self) -> None:

        pass

    def test_posterior__p_f__l_f(self):

        prior = 0.3
        likelihood = 0.8
        expected = (0.3 * 0.8) / ((0.3 * 0.8) + (0.7 * 0.2))

        bayes_rule = BayesRule(prior=prior, likelihood=likelihood)
        actual = bayes_rule.posterior()
        self.assertAlmostEqual(expected, actual, 10)

    def test_posterior__p_f__l_afm(self):

        prior = 0.3
        likelihood = Series({'$100': 0.8, '$200': 0.6})
        expected = Series({
            '$100': (0.3 * 0.8) / ((0.3 * 0.8) + 0.7 * 0.2),
            '$200': (0.3 * 0.6) / ((0.3 * 0.6) + 0.7 * 0.4)
        })

        bayes_rule = BayesRule(prior=prior, likelihood=likelihood)
        actual = bayes_rule.posterior()
        self.assertIsInstance(actual, Series)
        self.assertListEqual(list(expected.keys()), list(actual.keys()))
        for key in expected.keys():
            self.assertAlmostEqual(expected[key], actual[key])
