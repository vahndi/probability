from unittest.case import TestCase

from pandas import Series, DataFrame

from probability.distributions import Gamma, GammaExponentialConjugate
from probability.distributions.conjugate.priors import VaguePrior


class TestGammaExponentialConjugate(TestCase):

    def setUp(self) -> None:

        self.series = Series([
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2
        ])
        self.n_series = len(self.series)
        self.sum_series = self.series.sum()
        self.exponential_data = DataFrame({
            'a': [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2],
            'b': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            'c': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
            'd': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 0.1, 0.2, 0.3],
        })

    def test_infer_posterior(self):

        expected = Gamma(
            alpha=VaguePrior.Gamma.alpha + self.n_series,
            beta=VaguePrior.Gamma.beta + self.sum_series
        )
        actual = GammaExponentialConjugate.infer_posterior(data=self.series)
        self.assertAlmostEqual(actual.alpha, expected.alpha, 10)
        self.assertAlmostEqual(actual.beta, expected.beta, 10)

    def test_infer_posteriors(self):

        g__c__1_1 = Gamma(alpha=3 + 0.001, beta=sum([0.1, 0.2, 0.3, 0.001]))
        g__d__1_1 = Gamma(alpha=3 + 0.001, beta=sum([0.4, 0.5, 0.6, 0.001]))
        g__c__2_1 = Gamma(alpha=3 + 0.001, beta=sum([0.4, 0.5, 0.6, 0.001]))
        g__d__2_1 = Gamma(alpha=3 + 0.001, beta=sum([0.7, 0.8, 0.9, 0.001]))
        g__c__1_2 = Gamma(alpha=3 + 0.001, beta=sum([0.7, 0.8, 0.9, 0.001]))
        g__d__1_2 = Gamma(alpha=3 + 0.001, beta=sum([1.0, 1.1, 1.2, 0.001]))
        g__c__2_2 = Gamma(alpha=3 + 0.001, beta=sum([1.0, 1.1, 1.2, 0.001]))
        g__d__2_2 = Gamma(alpha=3 + 0.001, beta=sum([0.1, 0.2, 0.3, 0.001]))

        expected = DataFrame(
            data=[
                (1, 1, 'c', g__c__1_1),
                (2, 1, 'c', g__c__2_1),
                (1, 2, 'c', g__c__1_2),
                (2, 2, 'c', g__c__2_2),
                (1, 1, 'd', g__d__1_1),
                (2, 1, 'd', g__d__2_1),
                (1, 2, 'd', g__d__1_2),
                (2, 2, 'd', g__d__2_2)
            ],
            columns=['a', 'b', 'prob_var', 'Gamma']
        )
        actual = GammaExponentialConjugate.infer_posteriors(
            data=self.exponential_data,
            prob_vars=['c', 'd'],
            cond_vars=['a', 'b']
        )
        for _, row in expected.iterrows():
            actual_gamma = actual.loc[
                (actual['a'] == row['a']) &
                (actual['b'] == row['b']) &
                (actual['prob_var'] == row['prob_var']),
                'Gamma'
            ].iloc[0]
            self.assertAlmostEqual(row['Gamma'].alpha, actual_gamma.alpha, 10)
            self.assertAlmostEqual(row['Gamma'].beta, actual_gamma.beta, 10)
