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

    def test_se_mean_dist_and_ci(self):
        from probability.distributions.continuous.normal import Normal
        from probability.distributions.continuous.students_t import StudentsT
        from probability.distributions.data.interval import Interval

        s = Series([2.0, 4.0, 6.0, 8.0])
        r = Ratio(s)
        iv = Interval(s)
        c = Count(Series([2, 4, 6, 8]))
        b = Boolean(Series([True, False, True, False]))

        expected_se = float(s.sem())
        for dist in (r, iv, c, b):
            self.assertIsInstance(dist, NumericDataDistribution)
            self.assertAlmostEqual( dist.data.sem(), dist.se(), 10)
            self.assertAlmostEqual(dist.se(), dist.sem(), 10)

        norm_dist = r.mean_dist(use_t=False)
        self.assertIsInstance(norm_dist, Normal)
        self.assertAlmostEqual(5.0, norm_dist.mu, 10)
        self.assertAlmostEqual(expected_se, norm_dist.sigma, 10)

        t_dist = r.mean_dist(use_t=True)
        self.assertIsInstance(t_dist, StudentsT)
        self.assertEqual(3, t_dist.nu)
        self.assertAlmostEqual(5.0, t_dist.mu, 10)
        self.assertAlmostEqual(expected_se, t_dist.sigma, 10)

        ci_norm = r.ci(0.95, use_t=False)
        ci_t = r.ci(0.95, use_t=True)
        self.assertAlmostEqual(5.0 - 1.959963984540054 * expected_se, ci_norm[0], 6)
        self.assertAlmostEqual(5.0 + 1.959963984540054 * expected_se, ci_norm[1], 6)
        # t-interval with df=3 is wider than normal interval
        self.assertLess(ci_t[0], ci_norm[0])
        self.assertGreater(ci_t[1], ci_norm[1])

        # Series aggregates (e.g. RatioSeries, BooleanSeries)
        arm = Nominal(Series(['ctrl', 'ctrl', 'trat', 'trat']))
        rs = r | arm
        bs = b | arm
        self.assertEqual(2, len(rs.se()))
        self.assertEqual(2, len(bs.sem()))
        ci_df = rs.ci(0.95)
        self.assertEqual(['lower', 'upper'], list(ci_df.columns))
        self.assertEqual(['ctrl', 'trat'], list(ci_df.index))

    def test_hypothesis_testing_and_sample_size(self):
        from probability.distributions.continuous.normal import Normal
        from probability.distributions.continuous.students_t import StudentsT

        # Synthetic short_email vs no_email conversion data (like Chapter 2)
        short = Boolean(Series([True] * 15 + [False] * 105))  # n=120, mean=0.125
        no_email = Boolean(Series([True] * 4 + [False] * 90))  # n=94, mean=4/94

        diff_dist = short.mean_diff(no_email)
        self.assertIsInstance(diff_dist, Normal)
        self.assertEqual(short.mean_dist() - no_email.mean_dist(), diff_dist)

        diff_t_dist = short.mean_diff(no_email, use_t=True)
        self.assertIsInstance(diff_t_dist, StudentsT)

        # Shifted null difference (e.g., H0: diff = 0.01)
        ci_0 = short.mean_diff_ci(no_email, percent=0.95, null_diff=0.0)
        ci_shifted = short.mean_diff_ci(no_email, percent=0.95, null_diff=0.01)
        self.assertAlmostEqual(ci_0[0] - 0.01, ci_shifted[0], 10)
        self.assertAlmostEqual(ci_0[1] - 0.01, ci_shifted[1], 10)

        # Test statistic and p-value
        expected_se = (short.se() ** 2 + no_email.se() ** 2) ** 0.5
        expected_t = (short.mean() - no_email.mean()) / expected_se
        self.assertAlmostEqual(expected_t, short.t_stat(no_email), 10)

        p_two = short.p_value(no_email, two_sided=True)
        p_one = short.p_value(no_email, two_sided=False)
        self.assertAlmostEqual(p_two, 2.0 * p_one, 10)
        self.assertLess(p_two, 0.05)

        # Sample size calculation (exact and rule-of-16)
        n_16 = no_email.sample_size_for_effect(mde=0.08, approx_16=True)
        n_exact = no_email.sample_size_for_effect(mde=0.08, alpha=0.05, power=0.80)
        self.assertEqual(int(__import__('math').ceil(16 * (no_email.std() / 0.08) ** 2)), n_16)
        self.assertTrue(abs(n_16 - n_exact) <= 20)

    def test_covariate_balance_normalized_diff(self):
        # Chapter 2 covariate balance formula: (mu_k - mu_ctrl) / sqrt((var_k + var_ctrl) / 2)
        arm = Nominal(Series(['no_email', 'no_email', 'no_email', 'short', 'short', 'long', 'long']))
        age = Ratio(Series([20.0, 22.0, 24.0, 25.0, 27.0, 19.0, 21.0], name='age'))
        gender = Boolean(Series([True, False, True, False, True, False, False], name='gender'))

        no_email_age = age | arm.keep('no_email')
        short_age = age | arm.keep('short')
        expected_norm_diff = (
            (short_age.mean() - no_email_age.mean()) /
            ((short_age.var() + no_email_age.var()) / 2.0) ** 0.5
        )
        self.assertAlmostEqual(expected_norm_diff, short_age.normalized_diff(no_email_age), 10)
        self.assertAlmostEqual(expected_norm_diff, short_age.cohens_d(no_email_age, pooled=False), 10)
        self.assertNotAlmostEqual(
            short_age.cohens_d(no_email_age, pooled=True),
            short_age.cohens_d(no_email_age, pooled=False),
            3
        )

        # Series and conditional balance tables
        age_by_arm = age | arm
        norm_diffs = age_by_arm.normalized_diff(control='no_email')
        self.assertAlmostEqual(0.0, norm_diffs['no_email'], 10)
        self.assertAlmostEqual(expected_norm_diff, norm_diffs['short'], 10)

        cond_mat = age.conditional_normalized_diff(arm)
        self.assertAlmostEqual(expected_norm_diff, cond_mat.loc['no_email', 'short'], 10)

        # Boolean also supports cohens_d and normalized_diff
        gender_by_arm = gender | arm
        gender_diffs = gender_by_arm.normalized_diff(control='no_email')
        self.assertAlmostEqual(0.0, gender_diffs['no_email'], 10)
        self.assertEqual(3, len(gender_diffs))

    def test_quantiles_and_boolean_indicators(self):
        # Chapter 2 ENEM scores workflow: top/bottom quantile indicators and outlier trimming
        scores = Ratio(Series(range(1, 101), dtype=float, name='avg_score'))
        students = Count(Series(range(10, 110), name='number_of_students'))

        self.assertAlmostEqual(99.01, scores.quantile(0.99), 2)

        top_school = scores.ge_quantile(0.99)
        bottom_school = scores.le_quantile(0.01)
        self.assertIsInstance(top_school, Boolean)
        self.assertEqual(1, (top_school.data == True).sum())
        self.assertEqual(1, (bottom_school.data == True).sum())

        extreme_school = top_school.logical_or(bottom_school)
        middle_school = ~extreme_school
        self.assertEqual(2, (extreme_school.data == True).sum())
        self.assertEqual(98, (middle_school.data == True).sum())

        trimmed_students = students.between_quantiles(0.0, 0.98, inclusive='left')
        self.assertEqual(98, len(trimmed_students))

        students_by_top = students | top_school
        self.assertEqual(99, len(students_by_top[False]))
        self.assertEqual(1, len(students_by_top[True]))

        # Also works when a split category is empty after outlier trimming
        trimmed_by_top = trimmed_students | top_school
        self.assertEqual(98, len(trimmed_by_top[False]))
        self.assertEqual(0, len(trimmed_by_top[True]))








