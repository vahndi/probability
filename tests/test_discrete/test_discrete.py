from unittest.case import TestCase

from pandas import Series, DataFrame

from probability.discrete import Conditional
from probability.discrete.discrete import Discrete


class TestDiscrete(TestCase):

    def setUp(self) -> None:

        pass

    def test_from_counts__1_var__vars_on_index(self):

        counts = Series({
            'a': 1,
            'b': 2,
            'c': 3
        })
        counts.index.name = 'abc'
        discrete = Discrete.from_counts(counts)
        self.assertEqual(['abc'], discrete.variables)
        self.assertEqual({'abc': ['a', 'b', 'c']},
                         discrete.states)
        self.assertEqual(1 / 6, discrete.p(abc='a'))
        self.assertEqual(2 / 6, discrete.p(abc='b'))
        self.assertEqual(3 / 6, discrete.p(abc='c'))

    def test_from_counts__1_var__vars_as_arg(self):

        counts = Series({
            'a': 1,
            'b': 2,
            'c': 3
        })
        discrete = Discrete.from_counts(counts, variables='abc')
        self.assertEqual(['abc'], discrete.variables)
        self.assertEqual({'abc': ['a', 'b', 'c']}, discrete.states)
        self.assertEqual(1 / 6, discrete.p(abc='a'))
        self.assertEqual(2 / 6, discrete.p(abc='b'))
        self.assertEqual(3 / 6, discrete.p(abc='c'))

    def test_from_counts__1_var__states_as_arg(self):

        counts = Series({
            'a': 1,
            'b': 2,
            'c': 3
        })
        counts.index.name = 'abc'
        discrete = Discrete.from_counts(
            counts, states=['a', 'b', 'c', 'd']
        )
        self.assertEqual(['abc'], discrete.variables)
        self.assertEqual({'abc': ['a', 'b', 'c', 'd']},
                         discrete.states)
        self.assertEqual(1 / 6, discrete.p(abc='a'))
        self.assertEqual(2 / 6, discrete.p(abc='b'))
        self.assertEqual(3 / 6, discrete.p(abc='c'))
        self.assertEqual(0, discrete.p(abc='d'))

    def test_from_counts__2_vars__vars_on_index(self):

        counts = Series({
            ('a', 'b'): 1,
            ('c', 'd'): 2,
            ('e', 'f'): 3
        })
        counts.index.names = ['ace', 'bdf']
        discrete = Discrete.from_counts(counts)
        self.assertEqual(['ace', 'bdf'], discrete.variables)
        self.assertEqual({'ace': ['a', 'c', 'e'], 'bdf': ['b', 'd', 'f']},
                         discrete.states)
        self.assertEqual(1 / 6, discrete.p(ace='a', bdf='b'))
        self.assertEqual(2 / 6, discrete.p(ace='c', bdf='d'))
        self.assertEqual(3 / 6, discrete.p(ace='e', bdf='f'))

    def test_from_counts__2_vars__vars_as_arg(self):

        counts = Series({
            ('a', 'b'): 1,
            ('c', 'd'): 2,
            ('e', 'f'): 3
        })
        discrete = Discrete.from_counts(counts, variables=['ace', 'bdf'])
        self.assertEqual(['ace', 'bdf'], discrete.variables)
        self.assertEqual({'ace': ['a', 'c', 'e'], 'bdf': ['b', 'd', 'f']},
                         discrete.states)
        self.assertEqual(1 / 6, discrete.p(ace='a', bdf='b'))
        self.assertEqual(2 / 6, discrete.p(ace='c', bdf='d'))
        self.assertEqual(3 / 6, discrete.p(ace='e', bdf='f'))

    def test_from_counts__2_vars__states_as_arg(self):

        counts = Series({
            ('a', 'b'): 1,
            ('c', 'd'): 2,
            ('e', 'f'): 3
        })
        counts.index.names = ['ace', 'bdf']
        states = {'ace': ['a', 'c', 'e', 'g'],
                  'bdf': ['b', 'd', 'f', 'h']}
        discrete = Discrete.from_counts(
            counts,
            states=states
        )
        self.assertEqual(['ace', 'bdf'], discrete.variables)
        self.assertEqual(states, discrete.states)
        self.assertEqual(1 / 6, discrete.p(ace='a', bdf='b'))
        self.assertEqual(2 / 6, discrete.p(ace='c', bdf='d'))
        self.assertEqual(3 / 6, discrete.p(ace='e', bdf='f'))

    def test_from_observations__1_var(self):

        observations = DataFrame({
            'ace': ['a', 'c', 'c', 'e', 'e', 'e'],
        })
        discrete = Discrete.from_observations(observations)
        self.assertEqual(['ace'], discrete.variables)
        self.assertEqual({'ace': ['a', 'c', 'e']}, discrete.states)
        self.assertEqual(1 / 6, discrete.p(ace='a'))
        self.assertEqual(2 / 6, discrete.p(ace='c'))
        self.assertEqual(3 / 6, discrete.p(ace='e'))

    def test_from_observations__1_var__replace_vars(self):

        observations = DataFrame({
            'ace': ['a', 'c', 'c', 'e', 'e', 'e'],
        })
        discrete = Discrete.from_observations(
            observations, variables='ACE'
        )
        self.assertEqual(['ACE'], discrete.variables)
        self.assertEqual({'ACE': ['a', 'c', 'e']}, discrete.states)
        self.assertEqual(1 / 6, discrete.p(ACE='a'))
        self.assertEqual(2 / 6, discrete.p(ACE='c'))
        self.assertEqual(3 / 6, discrete.p(ACE='e'))

    def test_from_observations__1_var__extra_states(self):

        observations = DataFrame({
            'ace': ['a', 'c', 'c', 'e', 'e', 'e'],
        })
        discrete = Discrete.from_observations(
            observations, states=['a', 'c', 'e', 'g']
        )
        self.assertEqual(['ace'], discrete.variables)
        self.assertEqual({'ace': ['a', 'c', 'e', 'g']}, discrete.states)
        self.assertEqual(1 / 6, discrete.p(ace='a'))
        self.assertEqual(2 / 6, discrete.p(ace='c'))
        self.assertEqual(3 / 6, discrete.p(ace='e'))
        self.assertEqual(0, discrete.p(ace='g'))

    def test_from_observations__2_vars(self):

        observations = DataFrame({
            'ace': ['a', 'c', 'c', 'e', 'e', 'e'],
            'bdf': ['b', 'd', 'd', 'f', 'f', 'f']
        })
        discrete = Discrete.from_observations(observations)
        self.assertEqual(['ace', 'bdf'], discrete.variables)
        self.assertEqual({'ace': ['a', 'c', 'e'],
                          'bdf': ['b', 'd', 'f']},
                         discrete.states)
        self.assertEqual(1 / 6, discrete.p(ace='a', bdf='b'))
        self.assertEqual(2 / 6, discrete.p(ace='c', bdf='d'))
        self.assertEqual(3 / 6, discrete.p(ace='e', bdf='f'))

    def test_from_observations__2_vars__replace_vars(self):

        observations = DataFrame({
            'ace': ['a', 'c', 'c', 'e', 'e', 'e'],
            'bdf': ['b', 'd', 'd', 'f', 'f', 'f']
        })
        discrete = Discrete.from_observations(
            observations, variables=['ACE', 'BDF']
        )
        self.assertEqual(['ACE', 'BDF'], discrete.variables)
        self.assertEqual({'ACE': ['a', 'c', 'e'],
                          'BDF': ['b', 'd', 'f']},
                         discrete.states)
        self.assertEqual(1 / 6, discrete.p(ACE='a', BDF='b'))
        self.assertEqual(2 / 6, discrete.p(ACE='c', BDF='d'))
        self.assertEqual(3 / 6, discrete.p(ACE='e', BDF='f'))

    def test_from_observations__2_vars__extra_states(self):

        observations = DataFrame({
            'ace': ['a', 'c', 'c', 'e', 'e', 'e'],
            'bdf': ['b', 'd', 'd', 'f', 'f', 'f']
        })
        states = {
            'ace': ['a', 'c', 'e', 'g'],
            'bdf': ['b', 'd', 'f', 'h']
        }
        discrete = Discrete.from_observations(
            observations, states=states
        )
        self.assertEqual(['ace', 'bdf'], discrete.variables)
        self.assertEqual(states, discrete.states)
        self.assertEqual(1 / 6, discrete.p(ace='a', bdf='b'))
        self.assertEqual(2 / 6, discrete.p(ace='c', bdf='d'))
        self.assertEqual(3 / 6, discrete.p(ace='e', bdf='f'))

    def test_from_probs__with_dict(self):

        bowl = Discrete.from_probs({
            'bowl 1': 0.5, 'bowl 2': 0.5},
            variables=['bowl']
        )
        self.assertIsInstance(bowl, Discrete)
        mix_1994 = Discrete.from_probs({
            'brown': 0.3, 'yellow': 0.2, 'red': 0.2,
            'green': 0.1, 'orange': 0.1, 'tan': 0.1
        }, variables='color')
        self.assertIsInstance(mix_1994, Discrete)

    def test_conditional_all_variables(self):

        expected = Discrete.binary(0, 'A_xor_B').data
        xor = Conditional.binary_from_probs(
            data={
                (0, 0): 0,
                (0, 1): 1,
                (1, 0): 1,
                (1, 1): 0,
            },
            joint_variable='A_xor_B',
            conditional_variables=['A', 'B']
        )
        actual = xor.given(A=1, B=1).data
        self.assertTrue(expected.equals(actual))

    def test_conditional_one_variable(self):

        expected = Conditional.binary_from_probs(
            data={
                0: 0,
                1: 1,
            },
            joint_variable='A_xor_B',
            conditional_variables='B'
        ).data
        xor = Conditional.binary_from_probs(
            data={
                (0, 0): 0,
                (0, 1): 1,
                (1, 0): 1,
                (1, 1): 0,
            },
            joint_variable='A_xor_B',
            conditional_variables=['A', 'B']
        )
        actual = xor.given(A=0).data
        self.assertTrue(expected.equals(actual))

    def test_mode_categorical(self):

        counts = Series({
            'a': 1, 'c': 2, 'e': 3
        })
        discrete = Discrete.from_counts(counts, variables='x')
        self.assertEqual('e', discrete.mode())

    def test_mode_numeric(self):

        discrete = Discrete.from_probs(
            data={0: 0.7, 1000: 0.2, 2000: 0.1},
            variables='a'
        )
        self.assertEqual(0, discrete.mode())

    def test_mode_multi_categorical(self):

        counts = Series({
            ('a', 'b'): 1,
            ('c', 'd'): 2,
            ('e', 'f'): 3
        })
        discrete = Discrete.from_counts(counts, variables=['ace', 'bdf'])
        self.assertEqual(('e', 'f'), discrete.mode())

    def test_mean_numeric(self):

        discrete = Discrete.from_probs(
            data={0: 0.7, 1000: 0.2, 2000: 0.1},
            variables='a'
        )
        self.assertAlmostEqual(
            0 * 0.7 + 1000 * 0.2 + 2000 * 0.1,
            discrete.mean()
        )

    def test_mean_categorical(self):

        counts = Series({
            'a': 1, 'c': 2, 'e': 3
        })
        discrete = Discrete.from_counts(counts, variables='x')
        self.assertRaises(TypeError, discrete.mean)

    def test_min_numeric(self):

        discrete = Discrete.from_probs(
            data={0: 0.7, 1000: 0.2, 2000: 0.1},
            variables='a'
        )
        self.assertAlmostEqual(0, discrete.min())
        discrete_2 = Discrete.from_probs(
            data={0: 0.0, 1000: 0.2, 2000: 0.8},
            variables='a'
        )
        self.assertAlmostEqual(1000, discrete_2.min())

    def test_min_categorical(self):

        discrete = Discrete.from_probs(
            data={'a': 0.7, 'b': 0.2, 'c': 0.1},
            variables='x'
        )
        self.assertRaises(TypeError, discrete.min)

    def test_max_numeric(self):

        discrete = Discrete.from_probs(
            data={0: 0.7, 1000: 0.2, 2000: 0.1},
            variables='a'
        )
        self.assertAlmostEqual(2000, discrete.max())
        discrete_2 = Discrete.from_probs(
            data={0: 0.7, 1000: 0.3, 2000: 0},
            variables='a'
        )
        self.assertAlmostEqual(1000, discrete_2.max())

    def test_max_categorical(self):

        discrete = Discrete.from_probs(
            data={'a': 0.7, 'b': 0.2, 'c': 0.1},
            variables='x'
        )
        self.assertRaises(TypeError, discrete.max)
