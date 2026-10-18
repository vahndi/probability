from unittest.case import TestCase
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from probability.discrete import Discrete, Conditional, BayesianNetwork


class TestBayesianNetwork(TestCase):

    def setUp(self) -> None:
        # Butler, Maid, Knife from BRML Chapter 1
        self.butler = Discrete.from_probs({'yes': 0.6, 'no': 0.4}, variables='butler')
        self.maid = Discrete.from_probs({'yes': 0.2, 'no': 0.8}, variables='maid')
        self.knife = Conditional.from_probs({
            ('yes', 'no', 'no'): 0.3,
            ('yes', 'no', 'yes'): 0.2,
            ('yes', 'yes', 'no'): 0.6,
            ('yes', 'yes', 'yes'): 0.1,
            ('no', 'no', 'no'): 0.7,
            ('no', 'no', 'yes'): 0.8,
            ('no', 'yes', 'no'): 0.4,
            ('no', 'yes', 'yes'): 0.9,
        }, joint_variables='knife', conditional_variables=['butler', 'maid'])

        self.crime_net = BayesianNetwork([self.butler, self.maid, self.knife])

    def test_network_structure(self):
        self.assertCountEqual(self.crime_net.nodes, ['butler', 'maid', 'knife'])
        self.assertCountEqual(self.crime_net.parents('knife'), ['butler', 'maid'])
        self.assertEqual(self.crime_net.parents('butler'), [])
        self.assertCountEqual(self.crime_net.children('butler'), ['knife'])
        self.assertCountEqual(self.crime_net.children('maid'), ['knife'])

    def test_d_separation(self):
        # Butler and Maid are marginally independent (collider at Knife)
        self.assertTrue(self.crime_net.is_independent('butler', 'maid'))
        self.assertTrue(self.crime_net.d_separated('butler', 'maid'))

        # Butler and Maid are dependent given Knife (explaining away)
        self.assertFalse(self.crime_net.is_independent('butler', 'maid', given='knife'))

    def test_cycle_detection(self):
        net = BayesianNetwork()
        a = Discrete.from_probs({'t': 0.5, 'f': 0.5}, variables='A')
        b_given_a = Conditional.from_probs({
            ('t', 't'): 0.8, ('t', 'f'): 0.2,
            ('f', 't'): 0.2, ('f', 'f'): 0.8,
        }, joint_variables='B', conditional_variables='A')
        net.add_node(a)
        net.add_node(b_given_a)

        # Creating a cycle B -> A should raise ValueError
        a_given_b = Conditional.from_probs({
            ('t', 't'): 0.8, ('t', 'f'): 0.2,
            ('f', 't'): 0.2, ('f', 'f'): 0.8,
        }, joint_variables='A', conditional_variables='B')
        with self.assertRaises(ValueError):
            net.add_node(a_given_b)

    def test_variable_elimination_query(self):
        # Query P(butler | knife='yes')
        post = self.crime_net.query(target='butler', evidence={'knife': 'yes'})
        self.assertIsInstance(post, Discrete)
        # Expected from BRML Ch 1 manual calculation: ~0.728155
        p_butler_yes = post.p(butler='yes')
        self.assertAlmostEqual(p_butler_yes, 0.7281553398058251, places=5)

    def test_p_convenience_method(self):
        p1 = self.crime_net.p(butler='yes', given={'knife': 'yes'})
        p2 = self.crime_net.p('butler', 'yes', given={'knife': 'yes'})
        p3 = self.crime_net.p(butler='yes', knife='yes')
        self.assertAlmostEqual(p1, 0.7281553398058251, places=5)
        self.assertEqual(p1, p2)
        self.assertEqual(p1, p3)

    def test_joint_distribution(self):
        joint = self.crime_net.joint()
        self.assertIsInstance(joint, Discrete)
        self.assertCountEqual(joint.variables, ['butler', 'maid', 'knife'])
        self.assertAlmostEqual(joint.data.sum(), 1.0, places=6)
        # Check P(butler=yes, maid=no, knife=yes) = 0.6 * 0.8 * 0.6 = 0.288
        p_val = joint.p(butler='yes', maid='no', knife='yes')
        self.assertAlmostEqual(p_val, 0.288, places=6)

    def test_classic_alarm_network(self):
        """
        Classic Pearl / Russell & Norvig / Barber Alarm Network:
        B: Burglary, E: Earthquake, A: Alarm, J: JohnCalls, M: MaryCalls
        """
        b = Discrete.from_probs({'t': 0.001, 'f': 0.999}, variables='B')
        e = Discrete.from_probs({'t': 0.002, 'f': 0.998}, variables='E')
        a = Conditional.from_probs({
            ('t', 't', 't'): 0.95, ('f', 't', 't'): 0.05,
            ('t', 't', 'f'): 0.94, ('f', 't', 'f'): 0.06,
            ('t', 'f', 't'): 0.29, ('f', 'f', 't'): 0.71,
            ('t', 'f', 'f'): 0.001, ('f', 'f', 'f'): 0.999,
        }, joint_variables='A', conditional_variables=['B', 'E'])
        j = Conditional.from_probs({
            ('t', 't'): 0.90, ('f', 't'): 0.10,
            ('t', 'f'): 0.05, ('f', 'f'): 0.95,
        }, joint_variables='J', conditional_variables='A')
        m = Conditional.from_probs({
            ('t', 't'): 0.70, ('f', 't'): 0.30,
            ('t', 'f'): 0.01, ('f', 'f'): 0.99,
        }, joint_variables='M', conditional_variables='A')

        alarm_net = BayesianNetwork([b, e, a, j, m])

        # Test d-separations:
        # J is conditionally independent of M given A
        self.assertTrue(alarm_net.is_independent('J', 'M', given='A'))
        # J is dependent on M without knowing A
        self.assertFalse(alarm_net.is_independent('J', 'M'))
        # B is independent of E given nothing
        self.assertTrue(alarm_net.is_independent('B', 'E'))
        # B is dependent on E given A (collider)
        self.assertFalse(alarm_net.is_independent('B', 'E', given='A'))

        # Query P(B | J=t, M=t)
        # Textbook result: P(B=t | J=t, M=t) ~ 0.28417
        post_b = alarm_net.query(target='B', evidence={'J': 't', 'M': 't'})
        p_b_given_jm = post_b.p(B='t')
        self.assertAlmostEqual(p_b_given_jm, 0.28417, places=4)

        # Query P(A | B=t, E=f) = 0.94
        post_a = alarm_net.query(target='A', evidence={'B': 't', 'E': 'f'})
        self.assertAlmostEqual(post_a.p(A='t'), 0.94, places=5)

    def test_draw(self):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax_out = self.crime_net.draw(ax=ax)
        self.assertIsNotNone(ax_out)
        plt.close(fig)
