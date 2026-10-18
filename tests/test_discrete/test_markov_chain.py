from unittest.case import TestCase
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from probability.discrete import Discrete, Conditional, MarkovChain


class TestMarkovChain(TestCase):

    def setUp(self) -> None:
        # 2-state Weather chain
        # Sunny -> Sunny: 0.8, Sunny -> Rainy: 0.2
        # Rainy -> Sunny: 0.4, Rainy -> Rainy: 0.6
        self.weather_transitions = {
            ('Sunny', 'Sunny'): 0.8,
            ('Sunny', 'Rainy'): 0.2,
            ('Rainy', 'Sunny'): 0.4,
            ('Rainy', 'Rainy'): 0.6,
        }
        self.weather_chain = MarkovChain(
            transitions=self.weather_transitions,
            initial_distribution={'Sunny': 1.0, 'Rainy': 0.0}
        )

    def test_initialization_forms(self):
        # 1. From dict
        mc1 = MarkovChain(self.weather_transitions)
        self.assertEqual(sorted(mc1.states), ['Rainy', 'Sunny'])

        # 2. From DataFrame
        df = pd.DataFrame([
            [0.8, 0.2],
            [0.4, 0.6]
        ], index=['Sunny', 'Rainy'], columns=['Sunny', 'Rainy'])
        mc2 = MarkovChain(df)
        self.assertAlmostEqual(mc2.transition('Sunny', 'Sunny'), 0.8)
        self.assertAlmostEqual(mc2.transition('Rainy', 'Rainy'), 0.6)

        # 3. From numpy array with state names
        arr = np.array([[0.8, 0.2], [0.4, 0.6]])
        mc3 = MarkovChain(arr, states=['Sunny', 'Rainy'])
        self.assertEqual(mc3.states, ['Sunny', 'Rainy'])

    def test_n_step_transitions(self):
        # n = 0 is identity
        t0 = self.weather_chain.n_step_transition_matrix(0)
        np.testing.assert_allclose(t0.values, np.eye(2))

        # n = 1 is original
        t1 = self.weather_chain.n_step_transition_matrix(1)
        np.testing.assert_allclose(t1.values, self.weather_chain.transition_matrix.values)

        # n = 2: T^2 = [[0.8, 0.2], [0.4, 0.6]] @ [[0.8, 0.2], [0.4, 0.6]]
        # Row 0: 0.8*0.8 + 0.2*0.4 = 0.64 + 0.08 = 0.72; 0.8*0.2 + 0.2*0.6 = 0.16 + 0.12 = 0.28
        t2 = self.weather_chain.n_step_transition_matrix(2)
        self.assertAlmostEqual(t2.loc['Sunny', 'Sunny'], 0.72)
        self.assertAlmostEqual(t2.loc['Sunny', 'Rainy'], 0.28)

        # Error on negative n
        with self.assertRaises(ValueError):
            self.weather_chain.n_step_transition_matrix(-1)

    def test_state_distribution(self):
        # Start in Sunny: [1.0, 0.0]
        # At step 1: [0.8, 0.2]
        dist1 = self.weather_chain.state_distribution(1)
        self.assertAlmostEqual(dist1.p(state_1='Sunny'), 0.8)
        self.assertAlmostEqual(dist1.p(state_1='Rainy'), 0.2)

        # At step 2: [0.72, 0.28]
        dist2 = self.weather_chain.state_distribution(2)
        self.assertAlmostEqual(dist2.p(state_2='Sunny'), 0.72)
        self.assertAlmostEqual(dist2.p(state_2='Rainy'), 0.28)

    def test_stationary_distribution(self):
        # Solve:
        # pi_S * 0.8 + pi_R * 0.4 = pi_S => 0.4 pi_R = 0.2 pi_S => pi_S = 2 pi_R
        # pi_S + pi_R = 1 => 3 pi_R = 1 => pi_R = 1/3, pi_S = 2/3
        stat = self.weather_chain.stationary_distribution()
        self.assertAlmostEqual(stat.p(state='Sunny'), 2.0 / 3.0, places=5)
        self.assertAlmostEqual(stat.p(state='Rainy'), 1.0 / 3.0, places=5)

    def test_absorbing_chain_gamblers_ruin(self):
        # 4 states: $0 (ruin), $1, $2, $3 (win)
        # Prob(win each bet) = 0.5
        # States 0 and 3 are absorbing
        transitions = pd.DataFrame([
            [1.0, 0.0, 0.0, 0.0],  # state 0: absorbing
            [0.5, 0.0, 0.5, 0.0],  # state 1
            [0.0, 0.5, 0.0, 0.5],  # state 2
            [0.0, 0.0, 0.0, 1.0],  # state 3: absorbing
        ], index=['0', '1', '2', '3'], columns=['0', '1', '2', '3'])

        chain = MarkovChain(transitions)
        self.assertTrue(chain.is_absorbing_chain())
        self.assertCountEqual(chain.absorbing_states, ['0', '3'])
        self.assertCountEqual(chain.transient_states, ['1', '2'])

        # Fundamental matrix N = (I - Q)^-1
        # Q = [[0.0, 0.5], [0.5, 0.0]]
        # I - Q = [[1.0, -0.5], [-0.5, 1.0]]
        # det = 1 - 0.25 = 0.75
        # inv = (1 / 0.75) * [[1.0, 0.5], [0.5, 1.0]] = [[4/3, 2/3], [2/3, 4/3]]
        N = chain.fundamental_matrix()
        self.assertAlmostEqual(N.loc['1', '1'], 4.0 / 3.0)
        self.assertAlmostEqual(N.loc['1', '2'], 2.0 / 3.0)
        self.assertAlmostEqual(N.loc['2', '1'], 2.0 / 3.0)
        self.assertAlmostEqual(N.loc['2', '2'], 4.0 / 3.0)

        # Expected time to absorption t = N * 1
        # t_1 = 4/3 + 2/3 = 2.0 steps
        # t_2 = 2/3 + 4/3 = 2.0 steps
        t1 = chain.expected_time_to_absorption('1')
        t2 = chain.expected_time_to_absorption('2')
        self.assertAlmostEqual(t1, 2.0)
        self.assertAlmostEqual(t2, 2.0)
        self.assertAlmostEqual(chain.expected_time_to_absorption('0'), 0.0)

        # Absorption probabilities B = N * R
        # R = [[0.5, 0.0], [0.0, 0.5]]
        # B = N @ R = [[2/3, 1/3], [1/3, 2/3]]
        B = chain.absorption_probabilities()
        self.assertAlmostEqual(B.loc['1', '0'], 2.0 / 3.0)
        self.assertAlmostEqual(B.loc['1', '3'], 1.0 / 3.0)
        self.assertAlmostEqual(B.loc['2', '0'], 1.0 / 3.0)
        self.assertAlmostEqual(B.loc['2', '3'], 2.0 / 3.0)

    def test_simulation(self):
        traj = self.weather_chain.simulate(n_steps=50, random_state=42)
        self.assertEqual(len(traj), 51)
        self.assertEqual(traj[0], 'Sunny')
        for s in traj:
            self.assertIn(s, ['Sunny', 'Rainy'])

    def test_to_bayesian_network(self):
        # Unroll for 3 steps
        bn = self.weather_chain.to_bayesian_network(n_steps=3)
        self.assertCountEqual(bn.nodes, ['S_0', 'S_1', 'S_2', 'S_3'])

        # Markov property check via d-separation: S_0 is independent of S_2 given S_1
        self.assertTrue(bn.is_independent('S_0', 'S_2', given='S_1'))
        # But dependent if S_1 is not given
        self.assertFalse(bn.is_independent('S_0', 'S_2'))

        # Query marginal at S_1: start Sunny -> P(S_1 = Sunny) = 0.8
        p_s1 = bn.query('S_1')
        self.assertAlmostEqual(p_s1.p(S_1='Sunny'), 0.8)

    def test_draw(self):
        fig, ax = plt.subplots()
        ret_ax = self.weather_chain.draw(ax=ax)
        self.assertIs(ret_ax, ax)
        plt.close(fig)
