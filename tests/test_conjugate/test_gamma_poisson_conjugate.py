from unittest.case import TestCase

from pandas import Series, DataFrame

from probability.distributions import Gamma
from probability.distributions.conjugate.gamma_poisson_conjugate import \
    GammaPoissonConjugate


class TestGammaPoissonConjugate(TestCase):

    def setUp(self) -> None:

        self.series = Series([
            0, 0, 0, 1, 0, 0, 1, 2, 0, 1, 2, 3
        ])
        self.n_series = len(self.series)
        self.k_series = self.series.sum()
        self.poisson_data = DataFrame({
            'a': [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2],
            'b': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            'c': [0, 0, 0, 1, 0, 0, 1, 2, 0, 1, 2, 3],
            'd': [1, 2, 3, 0, 0, 0, 1, 0, 0, 1, 2, 0],
        })

    def test_infer_posterior(self):

        expected = Gamma(alpha=0.001+self.k_series,
                         beta=0.001 + self.n_series)
        actual = GammaPoissonConjugate.infer_posterior(self.series)
        self.assertAlmostEqual(expected.alpha, actual.alpha, 10)
        self.assertAlmostEqual(expected.beta, actual.beta, 10)

    def test_infer_posteriors(self):

        g__c__1_1 = Gamma(0.001 + 0, 0.001 + 3)
        g__d__1_1 = Gamma(0.001 + 6, 0.001 + 3)
        g__c__2_1 = Gamma(0.001 + 1, 0.001 + 3)
        g__d__2_1 = Gamma(0.001 + 0, 0.001 + 3)
        g__c__1_2 = Gamma(0.001 + 3, 0.001 + 3)
        g__d__1_2 = Gamma(0.001 + 1, 0.001 + 3)
        g__c__2_2 = Gamma(0.001 + 6, 0.001 + 3)
        g__d__2_2 = Gamma(0.001 + 3, 0.001 + 3)

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
        actual = GammaPoissonConjugate.infer_posteriors(
            data=self.poisson_data,
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
