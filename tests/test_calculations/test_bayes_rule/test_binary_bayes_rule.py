from unittest.case import TestCase

from pandas import Series

from probability.calculations.bayes_rule.binary_bayes_rule import \
    BinaryBayesRule
from probability.distributions import Beta


class TestBinaryBayesRule(TestCase):

    def setUp(self) -> None:

        self.prior_float = 0.3
        self.prior_beta = Beta(1 + 3, 1 + 7)
        self.prior_float_map = Series({'$100': 0.3, '$200': 0.2})
        self.prior_beta_map = Series({'$100': Beta(1 + 3, 1 + 7),
                                      '$200': Beta(1 + 2, 1 + 8)})
        self.likelihood_float = 0.8
        self.likelihood_float_map = Series({'$100': 0.8, '$200': 0.6})
        self.likelihood_beta = Beta(1 + 8, 1 + 2)
        self.likelihood_beta_map = Series({
            '$100': Beta(1 + 8, 1 + 2),
            '$200': Beta(1 + 6, 1 + 4)
        })

    def test_posterior__p_f__l_f(self):

        expected = (0.3 * 0.8) / ((0.3 * 0.8) + (0.7 * 0.2))
        bayes_rule = BinaryBayesRule(prior=self.prior_float,
                                     likelihood=self.likelihood_float)
        actual = bayes_rule.posterior()
        self.assertAlmostEqual(expected, actual, 10)

    def test_posterior__p_f__l_fm(self):

        expected = Series({
            '$100': (0.3 * 0.8) / ((0.3 * 0.8) + 0.7 * 0.2),
            '$200': (0.3 * 0.6) / ((0.3 * 0.6) + 0.7 * 0.4)
        })
        bayes_rule = BinaryBayesRule(prior=self.prior_float,
                                     likelihood=self.likelihood_float_map)
        actual = bayes_rule.posterior()
        self.assertIsInstance(actual, Series)
        self.assertListEqual(list(expected.keys()), list(actual.keys()))
        for key in expected.keys():
            self.assertAlmostEqual(expected[key], actual[key])

    def test_posterior__p_f__l_b(self):

        posterior = BinaryBayesRule(prior=self.prior_float,
                                    likelihood=self.likelihood_beta).posterior()
        self.assertEqual(
            f'({str(self.prior_float)} * {str(self.likelihood_beta)}) / '
            f'(({str(self.prior_float)} * {str(self.likelihood_beta)}) + '
            f'({1 - self.prior_float} * (1 - {str(self.likelihood_beta)})))',
            posterior.name
        )
        output = posterior.output()
        self.assertAlmostEqual(output.mean(), 0.58, 2)
        self.assertAlmostEqual(output.std(), 0.15, 2)

    def test_posterior__p_f__l_bm(self):

        posterior = BinaryBayesRule(prior=self.prior_float,
                                    likelihood=self.likelihood_beta_map).posterior()
        for key, likelihood in self.likelihood_beta_map.items():
            self.assertEqual(
                f'({str(self.prior_float)} * {str(likelihood)}) / '
                f'(({str(self.prior_float)} * {str(likelihood)}) + '
                f'({1 - self.prior_float} * (1 - {str(likelihood)})))',
                posterior[key].name
            )

    def test_posterior__p_b__l_f(self):

        posterior = BinaryBayesRule(
            prior=self.prior_beta,
            likelihood=self.likelihood_float
        ).posterior()
        self.assertEqual(
            f'({str(self.prior_beta)} * {str(self.likelihood_float)}) / '
            f'(({str(self.prior_beta)} * {str(self.likelihood_float)}) + '
            f'((1 - {self.prior_beta}) * {str(1 - self.likelihood_float)}))',
            posterior.name
        )
        output = posterior.output()
        self.assertAlmostEqual(output.mean(), 0.64, 2)
        self.assertAlmostEqual(output.std(), 0.14, 2)

    def test_posterior__p_b__l_fm(self):

        posterior = BinaryBayesRule(
            prior=self.prior_beta,
            likelihood=self.likelihood_float_map
        ).posterior()
        for key, likelihood in self.likelihood_float_map.items():
            self.assertEqual(
                f'({str(self.prior_beta)} * {str(likelihood)}) / '
                f'(({str(self.prior_beta)} * {str(likelihood)}) + '
                f'((1 - {self.prior_beta}) * {str(1 - likelihood)}))',
                posterior[key].name
            )

    def test_posterior__p_b__l_b(self):

        posterior = BinaryBayesRule(
            prior=self.prior_beta,
            likelihood=self.likelihood_beta
        ).posterior()
        self.assertEqual(
            f'({str(self.prior_beta)} * {str(self.likelihood_beta)}) / '
            f'(({str(self.prior_beta)} * {str(self.likelihood_beta)}) + '
            f'((1 - {self.prior_beta}) * (1 - {str(self.likelihood_beta)})))',
            posterior.name
        )
        output = posterior.output()
        self.assertAlmostEqual(output.mean(), 0.6, 1)
        self.assertAlmostEqual(output.std(), 0.2, 1)

    def test_posterior__p_b__l_bm(self):

        posterior = BinaryBayesRule(
            prior=self.prior_beta,
            likelihood=self.likelihood_beta_map
        ).posterior()
        for key, likelihood in self.likelihood_beta_map.items():
            self.assertEqual(
                f'({str(self.prior_beta)} * {str(likelihood)}) / '
                f'(({str(self.prior_beta)} * {str(likelihood)}) + '
                f'((1 - {self.prior_beta}) * (1 - {str(likelihood)})))',
                posterior[key].name
            )

    def test_posterior__p_fm__l_f(self):

        posterior = BinaryBayesRule(
            prior=self.prior_float_map,
            likelihood=self.likelihood_float
        ).posterior()
        normalization = (
            (self.prior_float_map * self.likelihood_float) +
            ((1 - self.prior_float_map) * (1 - self.likelihood_float))
        )
        for key in self.prior_float_map.keys():
            self.assertEqual(
                self.prior_float_map[key] *
                self.likelihood_float /
                normalization[key],
                posterior[key]
            )

    def test_posterior__p_fm__l_fm(self):

        posterior = BinaryBayesRule(
            prior=self.prior_float_map,
            likelihood=self.likelihood_float_map
        ).posterior()
        for key in self.prior_float_map.keys():
            prior = self.prior_float_map[key]
            likelihood = self.likelihood_float_map[key]
            self.assertEqual(
                (prior * likelihood) /
                ((prior * likelihood) + ((1 - prior) * (1 - likelihood))),
                posterior[key]
            )

    def test_posterior__p_fm__l_b(self):

        prior = self.prior_float_map
        likelihood = self.likelihood_beta
        posterior = BinaryBayesRule(
            prior=prior,
            likelihood=likelihood
        ).posterior()
        for key in self.prior_float_map.keys():
            self.assertEqual(
                f'({prior[key]} * {str(likelihood)}) / '
                f'(({prior[key]} * {str(likelihood)}) + '
                f'({1 - prior[key]} * (1 - {str(likelihood)})))',
                posterior[key].name
            )

    def test_posterior__p_fm__l_bm(self):

        prior = self.prior_float_map
        likelihood = self.likelihood_beta_map
        posterior = BinaryBayesRule(
            prior=prior,
            likelihood=likelihood
        ).posterior()
        for key in self.prior_float_map.keys():
            self.assertEqual(
                f'({prior[key]} * {str(likelihood[key])}) / '
                f'(({prior[key]} * {str(likelihood[key])}) + '
                f'({1 - prior[key]} * (1 - {str(likelihood[key])})))',
                posterior[key].name
            )

    def test_posterior__p_bm__l_f(self):

        prior = self.prior_beta_map
        likelihood = self.likelihood_float
        posterior = BinaryBayesRule(
            prior=prior,
            likelihood=likelihood
        ).posterior()
        for key in self.prior_float_map.keys():
            self.assertEqual(
                f'({prior[key]} * {str(likelihood)}) / '
                f'(({prior[key]} * {str(likelihood)}) + '
                f'(({str(1 - prior[key])}) * {str(1 - likelihood)}))',
                posterior[key].name
            )

    def test_posterior__p_bm__l_fm(self):

        prior = self.prior_float_map
        likelihood = self.likelihood_beta_map
        posterior = BinaryBayesRule(
            prior=prior,
            likelihood=likelihood
        ).posterior()
        for key in self.prior_float_map.keys():
            self.assertEqual(
                f'({prior[key]} * {str(likelihood[key])}) / '
                f'(({prior[key]} * {str(likelihood[key])}) + '
                f'({1 - prior[key]} * (1 - {str(likelihood[key])})))',
                posterior[key].name
            )

    def test_posterior__p_bm__l_b(self):

        prior = self.prior_beta_map
        likelihood = self.likelihood_beta
        posterior = BinaryBayesRule(
            prior=prior,
            likelihood=likelihood
        ).posterior()
        for key in self.prior_beta_map.keys():
            self.assertEqual(
                f'({prior[key]} * {str(likelihood)}) / '
                f'(({prior[key]} * {str(likelihood)}) + '
                f'(({str(1 - prior[key])}) * (1 - {str(likelihood)})))',
                posterior[key].name
            )

    def test_posterior__p_bm__l_bm(self):

        prior = self.prior_beta_map
        likelihood = self.likelihood_beta_map
        posterior = BinaryBayesRule(
            prior=prior,
            likelihood=likelihood
        ).posterior()
        for key in self.prior_beta_map.keys():
            self.assertEqual(
                f'({prior[key]} * {str(likelihood[key])}) / '
                f'(({prior[key]} * {str(likelihood[key])}) + '
                f'(({str(1 - prior[key])}) * (1 - {str(likelihood[key])})))',
                posterior[key].name
            )
