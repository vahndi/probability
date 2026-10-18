from unittest.case import TestCase
from types import SimpleNamespace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from probability.distributions.data.ratio import Ratio


class TestRatioPosteriorSamples(TestCase):

    def setUp(self) -> None:
        np.random.seed(42)
        # Sample 1000 draws from N(10, 2)
        self.normal_samples = np.random.normal(loc=10.0, scale=2.0, size=1000)
        self.ratio = Ratio.from_samples(self.normal_samples, name='theta')

    def test_from_samples_input_types(self):
        # 1. 1D array
        r1 = Ratio.from_samples(np.array([1.0, 2.0, 3.0]), name='r1')
        self.assertEqual(len(r1), 3)
        self.assertEqual(r1.name, 'r1')

        # 2. 2D array (e.g. 4 chains x 250 draws)
        chains = np.ones((4, 250)) * 5.0
        r2 = Ratio.from_samples(chains)
        self.assertEqual(len(r2), 1000)
        self.assertAlmostEqual(r2.mean(), 5.0)

        # 3. pandas Series
        s = pd.Series([10.0, 20.0, 30.0], name='my_param')
        r3 = Ratio.from_samples(s)
        self.assertEqual(r3.name, 'my_param')
        self.assertAlmostEqual(r3.mean(), 20.0)

        # 4. list
        r4 = Ratio.from_samples([1.0, 2.0, 3.0])
        self.assertEqual(len(r4), 3)

    def test_from_samples_arviz_duck_typing(self):
        # Simulate an ArviZ InferenceData-like object with .posterior
        mock_idata = SimpleNamespace(
            posterior={
                'beta_treatment': np.random.normal(0.5, 0.1, size=(4, 500)),
                'intercept': np.random.normal(2.0, 0.2, size=(4, 500))
            }
        )

        # Multi-variable requires var_name
        with self.assertRaises(ValueError):
            Ratio.from_samples(mock_idata)

        # Specific var_name extracts properly
        r_beta = Ratio.from_samples(mock_idata, var_name='beta_treatment')
        self.assertEqual(len(r_beta), 2000)
        self.assertAlmostEqual(r_beta.mean(), 0.5, places=1)
        self.assertEqual(r_beta.name, 'beta_treatment')

        # Single variable InferenceData infers var_name automatically
        single_var_idata = SimpleNamespace(
            posterior={'mu': np.random.normal(100.0, 5.0, size=(2, 250))}
        )
        r_mu = Ratio.from_samples(single_var_idata)
        self.assertEqual(len(r_mu), 500)
        self.assertEqual(r_mu.name, 'mu')

    def test_hdi(self):
        # N(10, 2) has theoretical 95% interval [10 - 1.96*2, 10 + 1.96*2] = [6.08, 13.92]
        low, high = self.ratio.hdi(prob=0.95)
        self.assertAlmostEqual(low, 6.08, places=0)
        self.assertAlmostEqual(high, 13.92, places=0)
        self.assertLess(low, high)

        # Invalid prob
        with self.assertRaises(ValueError):
            self.ratio.hdi(prob=1.5)
        with self.assertRaises(ValueError):
            self.ratio.hdi(prob=-0.1)

    def test_rope_analysis(self):
        # Posterior centered at 10 with std 2
        # ROPE [9, 11] should contain ~38% of the mass for N(10, 2)
        p_in = self.ratio.prob_in_rope(9.0, 11.0)
        p_out = self.ratio.prob_outside_rope(9.0, 11.0)
        self.assertAlmostEqual(p_in + p_out, 1.0)
        self.assertGreater(p_in, 0.3)
        self.assertLess(p_in, 0.5)

        # ROPE [-1, 1] (null value = 0) is far away from posterior centered at 10
        self.assertEqual(self.ratio.prob_in_rope(-1.0, 1.0), 0.0)
        self.assertEqual(self.ratio.prob_outside_rope(-1.0, 1.0), 1.0)

    def test_rope_decision_rules(self):
        # 1. Reject Null: treatment effect is clearly positive, ROPE around 0 is [-0.1, 0.1]
        effect_samples = np.random.normal(5.0, 0.5, size=1000)
        r_effect = Ratio.from_samples(effect_samples)
        self.assertEqual(r_effect.rope_decision(-0.1, 0.1), 'reject_null')

        # 2. Accept Null: effect is practically zero, HDI is entirely within [-2.0, 2.0]
        null_samples = np.random.normal(0.0, 0.2, size=1000)
        r_null = Ratio.from_samples(null_samples)
        self.assertEqual(r_null.rope_decision(-2.0, 2.0), 'accept_null')

        # 3. Undecided: effect HDI overlaps the boundary of ROPE
        borderline_samples = np.random.normal(1.0, 1.0, size=1000)
        r_border = Ratio.from_samples(borderline_samples)
        self.assertEqual(r_border.rope_decision(0.5, 2.5), 'undecided')

    def test_plot_rope(self):
        fig, ax = plt.subplots()
        ret_ax = self.ratio.plot_rope(low=9.0, high=11.0, hdi_prob=0.95, ax=ax)
        self.assertIs(ret_ax, ax)
        plt.close(fig)
