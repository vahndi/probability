from unittest.case import TestCase

from examples.notebooks.cip.make_data import (
    load_cross_sell_email, load_xmas_sales, make_potential_outcomes_table
)
from probability.distributions import Boolean, Nominal, Normal, Ratio


class TestCausalInferenceChapters01And02(TestCase):

    def test_chapter_01_potential_outcomes_and_bias_equation(self):
        po_df = make_potential_outcomes_table(observed_only=False)
        Y0 = Ratio(po_df['y0'])
        Y1 = Ratio(po_df['y1'])
        T = Boolean(po_df['t'].astype(bool))

        po_df = po_df.assign(
            y=(po_df['t'] * po_df['y1'] + (1 - po_df['t']) * po_df['y0']).astype(int),
            te=Y1.data - Y0.data
        )
        TE = Ratio(po_df['te'])
        Y_obs = Ratio(po_df['y'])

        ate = TE.mean()
        att = (TE | T)[True].mean()
        bias = (Y0 | T)[True].mean() - (Y0 | T)[False].mean()
        assoc_diff = (Y_obs | T)[True].mean() - (Y_obs | T)[False].mean()

        self.assertAlmostEqual(65.0, ate, 10)
        self.assertAlmostEqual(250.0 / 3.0, att, 10)
        self.assertAlmostEqual(att + bias, assoc_diff, 10)


    def test_chapter_01_xmas_sales(self):
        data = load_xmas_sales()
        weekly_sales = Ratio(data['weekly_amount_sold'])
        on_sale = Boolean(data['is_on_sale'].astype(bool))

        sales_by_sale = weekly_sales | on_sale
        self.assertGreater(
            sales_by_sale[True].mean(),
            sales_by_sale[False].mean()
        )
        self.assertLess(
            sales_by_sale[True].p_value(sales_by_sale[False]),
            1e-10
        )

    def test_chapter_02_cross_sell_email(self):
        data = load_cross_sell_email()
        email = Nominal(data['cross_sell_email'])
        conversion = Boolean(data['conversion'].astype(bool))
        age = Ratio(data['age'])
        gender = Boolean(data['gender'].astype(bool))

        conv_by_email = conversion | email
        short_email = conv_by_email['short']
        long_email = conv_by_email['long']
        no_email = conv_by_email['no_email']

        self.assertEqual(120, len(short_email))
        self.assertEqual(109, len(long_email))
        self.assertEqual(94, len(no_email))

        self.assertAlmostEqual(0.030316953129541618, short_email.se(), 10)
        self.assertAlmostEqual(0.021946024609185506, long_email.se(), 10)

        # Covariate balance vs 'no_email'
        age_balance = (age | email).normalized_diff(control='no_email')
        gender_balance = (gender | email).normalized_diff(control='no_email')
        self.assertAlmostEqual(0.0, age_balance['no_email'], 10)
        self.assertAlmostEqual(0.0, gender_balance['no_email'], 10)
        self.assertLess(abs(age_balance['short']), 0.25)
        self.assertLess(abs(gender_balance['short']), 0.25)

        # Difference in means, t-stat, p-value, and sample size
        diff_dist = short_email.mean_diff(no_email)
        self.assertIsInstance(diff_dist, Normal)
        self.assertAlmostEqual(
            short_email.mean() - no_email.mean(),
            diff_dist.mu,
            10
        )
        self.assertLess(short_email.p_value(no_email, two_sided=True), 0.05)
        self.assertEqual(103, no_email.sample_size_for_effect(mde=0.08, approx_16=True))
