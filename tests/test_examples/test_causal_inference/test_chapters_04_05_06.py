from unittest.case import TestCase
import numpy as np

from examples.notebooks.cip.make_data import load_wage, load_management_training
from probability.distributions import Ratio, Boolean, StudentsT
from probability.models import LinearRegression, IPW, AIPW


class TestCausalInferenceChapters04To06(TestCase):

    def test_chapter_04_wage_regression_and_fwl(self):
        wage_df = load_wage().dropna(subset=['wage', 'educ', 'IQ', 'exper', 'tenure', 'age'])
        controls = ['educ', 'IQ', 'exper', 'tenure', 'age', 'married']

        # Controlled model with HC3
        model = LinearRegression.fit(
            X=wage_df[controls],
            y=wage_df['wage'],
            cov_type='HC3'
        )

        # Coefficient is positive return to education
        beta_educ = model.coefficient('educ')
        self.assertIsInstance(beta_educ, StudentsT)
        self.assertGreater(float(beta_educ.mu), 25.0)
        self.assertGreater(float(beta_educ > 0), 0.999)

        # FWL theorem holds
        tilde_educ, tilde_wage, fwl_bivariate = model.fwl_decompose('educ')
        beta_multi = float(model.params['educ'])
        beta_fwl = float(fwl_bivariate.params['educ_residualized'])
        self.assertAlmostEqual(beta_multi, beta_fwl, places=8)

        # Residualization reduces variance
        raw_wage = Ratio(wage_df['wage'])
        adj_wage = raw_wage.residualize(wage_df[['IQ', 'exper', 'tenure', 'age', 'married']], keep_mean=True)
        self.assertLess(adj_wage.std(), raw_wage.std())
        self.assertAlmostEqual(float(adj_wage.mean()), float(raw_wage.mean()), places=6)

    def test_chapter_05_06_management_training_ipw_and_aipw(self):
        mgt_df = load_management_training().dropna()
        covariate_cols = [
            'tenure', 'n_of_reports', 'gender', 'last_engagement_score',
            'department_score', 'department_size'
        ]

        # IPW ATE
        ipw = IPW.fit(
            treatment=mgt_df['intervention'],
            outcome=mgt_df['engagement_score'],
            covariates=mgt_df[covariate_cols],
            estimand='ATE'
        )
        self.assertGreater(ipw.estimate, 0.0)

        # Covariate balance
        bal = ipw.balance_summary()
        self.assertLess(np.mean(bal['abs_weighted']), np.mean(bal['abs_unweighted']))

        # AIPW Doubly Robust
        aipw = AIPW.fit(
            treatment=mgt_df['intervention'],
            outcome=mgt_df['engagement_score'],
            covariates=mgt_df[covariate_cols],
            estimand='ATE'
        )
        self.assertGreater(aipw.estimate, 0.0)
        self.assertLess(aipw.std_err, ipw.std_err)

        eff_dist = aipw.effect_distribution
        self.assertIsInstance(eff_dist, StudentsT)
        self.assertGreater(float(eff_dist > 0), 0.99)
