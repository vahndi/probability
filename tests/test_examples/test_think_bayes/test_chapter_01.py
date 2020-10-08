from unittest.case import TestCase

from pandas import DataFrame

from probability.calculations.bayes_rule import MultipleBayesRule
from probability.discrete.discrete import Discrete


class TestChapter01(TestCase):

    def setUp(self) -> None:

        # cookies
        self.bowl_1_and_chocolate = 0.125
        self.bowl_1_and_vanilla = 0.375
        self.bowl_2_and_chocolate = 0.25
        self.bowl_2_and_vanilla = 0.25
        cookie_data = TestChapter01.make_cookies_observations()
        self.cookies = Discrete.from_observations(cookie_data)
        self.vanilla = self.cookies.p(flavor='vanilla')
        self.vanilla__bowl_1 = self.cookies.given(
            bowl='bowl 1').p(flavor='vanilla')
        self.vanilla__bowl_2 = self.cookies.given(
            bowl='bowl 2').p(flavor='vanilla')
        self.bowl = Discrete.from_probs({
            'bowl 1': 0.5, 'bowl 2': 0.5},
            variables=['bowl']
        )
        self.bowl_1 = self.bowl.p(bowl='bowl 1')
        self.bowl_2 = self.bowl.p(bowl='bowl 2')

        # m & m's
        self.mix_1994 = Discrete.from_probs({
            'brown': 0.3, 'yellow': 0.2, 'red': 0.2,
            'green': 0.1, 'orange': 0.1, 'tan': 0.1
        }, variables='color')
        self.mix_1996 = Discrete.from_probs({
            'blue': 0.24, 'green': 0.2, 'orange': 0.16,
            'yellow': 0.14, 'red': 0.13, 'brown': 0.13
        }, variables='color')
        self.bag = Discrete.from_probs({1994: 0.5, 1996: 0.5}, variables='bag')

    @staticmethod
    def make_cookies_observations() -> DataFrame:

        return DataFrame({
            'bowl': ['bowl 1'] * 40 + ['bowl 2'] * 40,
            'flavor': (
                    ['vanilla'] * 30 + ['chocolate'] * 10 +
                    ['vanilla'] * 20 + ['chocolate'] * 20
            )
        })

    def test__01_03(self):

        data = self.cookies.data
        self.assertEqual(data.loc[('bowl 1', 'chocolate')],
                         self.bowl_1_and_chocolate)
        self.assertEqual(data.loc[('bowl 1', 'vanilla')],
                         self.bowl_1_and_vanilla)
        self.assertEqual(data.loc[('bowl 2', 'chocolate')],
                         self.bowl_2_and_chocolate)
        self.assertEqual(data.loc[('bowl 2', 'vanilla')],
                         self.bowl_2_and_vanilla)

    def test__01_04(self):

        self.assertEqual(self.bowl_1, 0.5)
        self.assertEqual(self.vanilla__bowl_1, 0.75)
        self.assertEqual(0.625, self.vanilla)
        bowl_1__vanilla = self.bowl_1 * self.vanilla__bowl_1 / self.vanilla
        self.assertEqual(0.6, bowl_1__vanilla)
        self.assertEqual(
            0.6, self.cookies.given(flavor='vanilla').p(bowl='bowl 1')
        )

    def test__01_05(self):

        self.assertEqual(self.bowl_2, 0.5)
        vanilla__bowl_2 = self.cookies.given(
            bowl='bowl 2').p(flavor='vanilla')
        self.assertEqual(0.5, vanilla__bowl_2)
        vanilla = (
            self.bowl_1 * self.vanilla__bowl_1 +
            self.bowl_2 * self.vanilla__bowl_2
        )
        self.assertEqual(0.625, vanilla)

    def test__01_06(self):

        yellow__1994 = self.mix_1994.p(color='yellow')
        yellow__1996 = self.mix_1996.p(color='yellow')
        green__1994 = self.mix_1994.p(color='green')
        green__1996 = self.mix_1996.p(color='green')
        likelihood_a = yellow__1994 * green__1996
        likelihood_b = yellow__1996 * green__1994
        prior_likelihood_a = self.bag.p(bag=1994) * likelihood_a
        prior_likelihood_b = self.bag.p(bag=1996) * likelihood_b
        evidence = prior_likelihood_a + prior_likelihood_b
        self.assertAlmostEqual(20 / 27, prior_likelihood_a / evidence, 10)
        self.assertAlmostEqual(7 / 27, prior_likelihood_b / evidence, 10)
        # using BayesRule
        priors = {'a': 0.5, 'b': 0.5}
        likelihoods = {
            'a': yellow__1994 * green__1996,
            'b': yellow__1996 * green__1994
        }
        bayes_rule = MultipleBayesRule(
            prior=priors, likelihood=likelihoods
        )
        posterior = bayes_rule.posterior()
        self.assertAlmostEqual(20 / 27, posterior['a'], 10)
        self.assertAlmostEqual(7 / 27, posterior['b'], 10)

    def test__01_07(self):

        priors = {'a': 1 / 3, 'b': 1 / 3, 'c': 1 / 3}
        likelihoods = {'a': 1 / 2, 'b': 0, 'c': 1}
        posterior = MultipleBayesRule(
            prior=priors, likelihood=likelihoods
        ).posterior()
        self.assertEqual(posterior['a'], 1 / 3)
        self.assertEqual(posterior['b'], 0)
        self.assertEqual(posterior['c'], 2 / 3)
        likelihoods = {'a': 1, 'b': 0, 'c': 1}
        posterior = MultipleBayesRule(
            prior=priors, likelihood=likelihoods
        ).posterior()
        self.assertEqual(posterior['a'], 1 / 2)
        self.assertEqual(posterior['b'], 0)
        self.assertEqual(posterior['c'], 1 / 2)
