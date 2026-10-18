from unittest.case import TestCase
import pandas as pd

from probability.discrete import MarkovChain, HMM


class TestChapter23(TestCase):

    def test_weather_markov_chain(self):
        weather_transitions = pd.DataFrame([
            [0.80, 0.20],
            [0.40, 0.60]
        ], index=['Sunny', 'Rainy'], columns=['Sunny', 'Rainy'])

        mc = MarkovChain(weather_transitions, initial_distribution={'Sunny': 1.0, 'Rainy': 0.0})
        stat = mc.stationary_distribution()
        self.assertAlmostEqual(stat.p(state='Sunny'), 2.0 / 3.0, places=4)
        self.assertAlmostEqual(stat.p(state='Rainy'), 1.0 / 3.0, places=4)

    def test_dishonest_casino_hmm(self):
        states = ['Fair', 'Loaded']
        emissions = [1, 2, 3, 4, 5, 6]

        transitions = pd.DataFrame([
            [0.95, 0.05],
            [0.10, 0.90]
        ], index=states, columns=states)

        emission_matrix = pd.DataFrame([
            [1.0 / 6] * 6,
            [0.10, 0.10, 0.10, 0.10, 0.10, 0.50]
        ], index=states, columns=emissions)

        hmm = HMM(
            states=states,
            emissions=emissions,
            transitions=transitions,
            emission_matrix=emission_matrix,
            initial_distribution={'Fair': 0.8, 'Loaded': 0.2}
        )

        obs = [1, 2, 3, 4, 5] * 3 + [6] * 15 + [1, 2, 3, 4, 5] * 3
        decoded, logp = hmm.decode(obs)
        self.assertEqual(decoded[0], 'Fair')
        self.assertEqual(decoded[20], 'Loaded')
        self.assertEqual(decoded[-1], 'Fair')
