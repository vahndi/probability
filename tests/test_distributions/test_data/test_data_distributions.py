from unittest.case import TestCase

from pandas import Series, CategoricalDtype

from probability.distributions.data.boolean import Boolean
from probability.distributions.data.count import Count
from probability.distributions.data.nominal import Nominal
from probability.distributions.data.ordinal import Ordinal
from probability.distributions.data.ratio import Ratio
from probability.distributions.protocols import (
    CategoricalDistribution, NumericDataDistribution, SupportsDataSeries
)


class TestDataDistributions(TestCase):

    def test_information_identities(self):

        x = Nominal(Series(['a', 'a', 'b', 'b', 'c', 'c', 'a', 'b']))
        y = Nominal(Series(['1', '1', '1', '2', '2', '2', '2', '1']))
        # H(X, Y) = H(Y) + H(X | Y)
        self.assertAlmostEqual(
            x.joint_entropy(y),
            y.entropy() + x.conditional_entropy(y),
            10
        )
        # I(X, Y) = H(X) - H(X | Y)
        self.assertAlmostEqual(
            x.mutual_information(y),
            x.entropy() - x.conditional_entropy(y),
            10
        )
        # NMI(X, X) == 1.0
        self.assertAlmostEqual(1.0, x.nmi(x, method='avg'), 10)
        self.assertAlmostEqual(1.0, x.nmi(x, method='max'), 10)

    def test_nmi_with_partial_index_overlap(self):

        x = Nominal(Series(['a', 'b'] * 10, index=range(20)))
        y = Nominal(Series(['a', 'b'] * 5, index=range(10)))
        self.assertAlmostEqual(1.0, x.nmi(y, method='avg'), 10)
        self.assertAlmostEqual(1.0, x.nmi(y, method='max'), 10)

    def test_ordinal_median_even_and_odd(self):

        cat_type = CategoricalDtype(
            categories=['low', 'medium', 'high'],
            ordered=True
        )
        odd_ord = Ordinal(Series(['low', 'medium', 'high'], dtype=cat_type))
        self.assertEqual('medium', odd_ord.median())

        even_ord = Ordinal(Series(['low', 'medium'], dtype=cat_type))
        self.assertEqual(('low', 'medium'), even_ord.median())

    def test_count_skellam_comparison(self):

        c1 = Count(Series([1, 2, 1, 2, 1, 2]))
        c2 = Count(Series([5, 6, 5, 6, 5, 6]))
        self.assertGreater(c1.probably_less_than(c2), 0.9)
        self.assertLess(c1.probably_greater_than(c2), 0.1)

    def test_ordinal_and_nominal_conversions(self):

        cat_type = CategoricalDtype(
            categories=['low', 'medium', 'high'],
            ordered=True
        )
        ord_dist = Ordinal(Series(['low', 'medium', 'high', 'high'], dtype=cat_type))
        bool_from_ord = ord_dist.as_boolean(false='low', true=['medium', 'high'])
        self.assertAlmostEqual(0.75, bool_from_ord.mean())

        ratio_from_ord = ord_dist.as_ratio({'low': 1.0, 'medium': 2.0, 'high': 3.0})
        self.assertAlmostEqual(2.25, ratio_from_ord.mean())

        nom_dist = Nominal(Series(['cat', 'dog', 'cat', 'bird']))
        bool_from_nom = nom_dist.as_boolean(false='bird', true=['cat', 'dog'])
        self.assertAlmostEqual(0.75, bool_from_nom.mean())

    def test_prob_comparisons_and_series_trends(self):

        c1 = Count(Series([1, 2, 1, 2, 1, 2]))
        c2 = Count(Series([8, 9, 8, 9, 8, 9]))
        self.assertEqual(1.0, c1 < c2)
        self.assertEqual(0.0, c1 > c2)
        self.assertEqual(0.0, c1 == c2)

        group = Nominal(Series(['g1'] * 3 + ['g2'] * 3 + ['g3'] * 3))
        counts = Count(Series([1, 1, 2, 4, 5, 5, 9, 10, 10]))
        cs = counts.split_by(group)
        inc_probs = cs.increase_probs()
        self.assertEqual(2, len(inc_probs))
        self.assertTrue((inc_probs > 0.8).all())
        p_inc = cs.p_increasing(n_iter=20)
        p_dec = cs.p_decreasing(n_iter=20)
        self.assertTrue(0.0 <= p_inc <= 1.0)
        self.assertTrue(0.0 <= p_dec <= 1.0)

    def test_protocols_and_sortable(self):

        nom = Nominal(Series(['a', 'b', 'a']))
        boolean = Boolean(Series([True, False, True]))
        ratio = Ratio(Series([1.0, 2.0, 3.0, 4.0]))

        self.assertIsInstance(nom, CategoricalDistribution)
        self.assertIsInstance(boolean, CategoricalDistribution)
        self.assertIsInstance(ratio, NumericDataDistribution)
        self.assertIsInstance(ratio.split_by( Nominal(Series(['x', 'x', 'y', 'y']))), SupportsDataSeries)

        self.assertEqual(2, len(ratio.upper(n=2)))
        self.assertEqual(3.5, ratio.upper(frac=0.5).mean())
        self.assertEqual(1.5, ratio.lower(frac=0.5).mean())

    def test_extended_protocols(self):
        from probability.calculations.mixins import (
            SupportsCalculation, SupportsOperator
        )
        from probability.calculations.operators import AddOperator
        from probability.distributions import (
            Beta, BetaBinomialConjugate, Binomial, Dirichlet
        )
        from probability.distributions.protocols import (
            SupportsConjugate, SupportsContinuousRV1d, SupportsDiscreteRV1d,
            SupportsPredictive, SupportsRV1d, SupportsRVSNd,
            SupportsResetDistribution
        )

        beta = Beta(2, 5)
        binom = Binomial(10, 0.5)
        dirichlet = Dirichlet([1, 2, 3])
        conj = BetaBinomialConjugate(alpha=2, beta=5, k=3, n=10)
        calc = beta + binom

        self.assertIsInstance(beta, SupportsRV1d)
        self.assertIsInstance(beta, SupportsContinuousRV1d)
        self.assertIsInstance(beta, SupportsResetDistribution)
        self.assertIsInstance(binom, SupportsRV1d)
        self.assertIsInstance(binom, SupportsDiscreteRV1d)
        self.assertIsInstance(dirichlet, SupportsRVSNd)
        self.assertIsInstance(conj, SupportsConjugate)
        self.assertIsInstance(conj, SupportsPredictive)
        self.assertIsInstance(calc, SupportsCalculation)
        self.assertIsInstance(AddOperator, SupportsOperator)

        # Verify protocol-based interoperability: Continuous/Discrete RV with
        # empirical NumericDataDistribution (SupportsRVS1d)
        ratio = Ratio(Series([0.1, 0.2, 0.3, 0.4]))
        combined = (beta + ratio).output(num_samples=50)
        self.assertEqual(50, len(combined))

        # Verify arithmetic between numeric empirical data distributions
        ratio_2 = Ratio(Series([1.0, 2.0, 3.0, 4.0]))
        ratio_sum = (ratio + ratio_2).output(num_samples=50)
        self.assertEqual(50, len(ratio_sum))
        ratio_quot = (ratio_2 / ratio).output(num_samples=50)
        self.assertEqual(50, len(ratio_quot))

        # Verify pipe (|) conditioning on empirical data distributions
        cats = Nominal(Series(['uk', 'uk', 'us', 'us']))
        split_series = ratio_2 | cats
        self.assertIsInstance(split_series, SupportsDataSeries)
        self.assertEqual(1.5, split_series['uk'].mean())
        self.assertEqual(3.5, split_series['us'].mean())

        uk_only = ratio_2 | cats.keep('uk')
        self.assertIsInstance(uk_only, Ratio)
        self.assertEqual(2, len(uk_only))
        self.assertEqual(1.5, uk_only.mean())

        gt_2 = ratio_2 | (ratio_2.data > 2.0)
        self.assertIsInstance(gt_2, Ratio)
        self.assertEqual(2, len(gt_2))
        self.assertEqual(3.5, gt_2.mean())



