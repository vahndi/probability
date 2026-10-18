from unittest.case import TestCase
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from probability.discrete import HMM


class TestHMM(TestCase):

    def setUp(self) -> None:
        # Classic Dishonest Casino model (Durbin et al. / BRML Ch 23)
        self.states = ['Fair', 'Loaded']
        self.emissions = [1, 2, 3, 4, 5, 6]

        self.transitions = pd.DataFrame([
            [0.95, 0.05],  # Fair -> Fair, Loaded
            [0.10, 0.90],  # Loaded -> Fair, Loaded
        ], index=self.states, columns=self.states)

        self.emission_matrix = pd.DataFrame([
            [1.0 / 6] * 6,                       # Fair: 1/6 for each
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.5],     # Loaded: 0.5 for '6'
        ], index=self.states, columns=self.emissions)

        self.initial_dist = {'Fair': 0.5, 'Loaded': 0.5}

        self.casino_hmm = HMM(
            states=self.states,
            emissions=self.emissions,
            transitions=self.transitions,
            emission_matrix=self.emission_matrix,
            initial_distribution=self.initial_dist
        )

    def test_properties(self):
        self.assertEqual(self.casino_hmm.states, ['Fair', 'Loaded'])
        self.assertEqual(self.casino_hmm.emissions, [1, 2, 3, 4, 5, 6])
        self.assertAlmostEqual(self.casino_hmm.transition_matrix.loc['Fair', 'Loaded'], 0.05)
        self.assertAlmostEqual(self.casino_hmm.emission_matrix.loc['Loaded', 6], 0.5)

    def test_likelihood_and_log_likelihood(self):
        obs = [1, 2, 6, 6, 6]
        log_lik = self.casino_hmm.log_likelihood(obs)
        lik = self.casino_hmm.likelihood(obs)

        self.assertLess(log_lik, 0.0)
        self.assertGreater(lik, 0.0)
        self.assertAlmostEqual(np.exp(log_lik), lik, places=6)

        # Empty observations
        self.assertEqual(self.casino_hmm.log_likelihood([]), 0.0)
        self.assertEqual(self.casino_hmm.likelihood([]), 1.0)

    def test_filtering_and_smoothing(self):
        # 10 fair rolls, 12 loaded sixes, 10 fair rolls
        obs = [1, 2, 3, 4, 5, 2, 1, 3, 4, 2] + [6] * 12 + [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]

        filtered = self.casino_hmm.filter(obs)
        self.assertEqual(filtered.shape, (len(obs), 2))
        np.testing.assert_allclose(filtered.sum(axis=1).values, 1.0)

        smoothed = self.casino_hmm.smooth(obs)
        self.assertEqual(smoothed.shape, (len(obs), 2))
        np.testing.assert_allclose(smoothed.sum(axis=1).values, 1.0)

        # Early in fair run, Fair should dominate
        self.assertGreater(smoothed.loc[5, 'Fair'], 0.7)

        # In middle of long run of sixes (around step 16), Loaded should strongly dominate
        self.assertGreater(smoothed.loc[16, 'Loaded'], 0.8)

        # At the end of final fair run, Fair should dominate
        self.assertGreater(smoothed.loc[len(obs) - 2, 'Fair'], 0.7)

    def test_viterbi_decoding(self):
        # Sequence of 15 fair rolls, 15 loaded sixes, 15 fair rolls
        obs = [1, 2, 3, 4, 5] * 3 + [6] * 15 + [1, 2, 3, 4, 5] * 3
        decoded_states, log_prob = self.casino_hmm.decode(obs)

        self.assertEqual(len(decoded_states), len(obs))
        self.assertLess(log_prob, 0.0)

        # Early rolls decode to 'Fair'
        self.assertEqual(decoded_states[0], 'Fair')
        self.assertEqual(decoded_states[5], 'Fair')

        # Middle block of sixes (steps 18-28) decodes to 'Loaded'
        self.assertEqual(decoded_states[20], 'Loaded')
        self.assertEqual(decoded_states[25], 'Loaded')

        # Final rolls decode to 'Fair'
        self.assertEqual(decoded_states[-1], 'Fair')
        self.assertEqual(decoded_states[-5], 'Fair')

    def test_sample(self):
        hidden, emiss = self.casino_hmm.sample(n_steps=30, random_state=42)
        self.assertEqual(len(hidden), 30)
        self.assertEqual(len(emiss), 30)
        for s in hidden:
            self.assertIn(s, self.states)
        for e in emiss:
            self.assertIn(e, self.emissions)

    def test_plot_states(self):
        obs = [1, 6, 6, 6, 2]
        fig, ax = plt.subplots()
        ret_ax = self.casino_hmm.plot_states(obs, smoothed=True, ax=ax)
        self.assertIs(ret_ax, ax)
        plt.close(fig)

    def test_to_bayesian_network(self):
        bn = self.casino_hmm.to_bayesian_network(n_steps=2)
        self.assertCountEqual(bn.nodes, ['Z_0', 'X_0', 'Z_1', 'X_1', 'Z_2', 'X_2'])

        # Check d-separation: X_0 is conditionally independent of X_1 given Z_0 (or Z_1)
        self.assertTrue(bn.is_independent('X_0', 'X_1', given=['Z_0']))
        # X_0 and X_1 are marginally dependent
        self.assertFalse(bn.is_independent('X_0', 'X_1'))

        # Exact query via BayesianNetwork: P(Z_1 | X_0=6, X_1=6)
        query_res = bn.query('Z_1', evidence={'X_0': 6, 'X_1': 6})
        # Two consecutive sixes should increase the posterior of Loaded
        self.assertGreater(query_res.p(Z_1='Loaded'), self.initial_dist['Loaded'])
