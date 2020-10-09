from itertools import product
from unittest.case import TestCase

from pandas import Series, Index, DataFrame

from probability.discrete.conditional import Conditional
from probability.discrete.discrete import Discrete


class TestChapter01(TestCase):

    def setUp(self) -> None:

        self.darts = Discrete(
            data=Series(
                index=Index(data=range(1, 21), name='region'),
                data=1 / 20
            ),
            variables='region', states=list(range(1, 21))
        )

    def test__1_1_1(self):

        darts = self.darts
        for region in range(1, 21):
            self.assertEqual(1 / 20, darts.p(region=region))

        self.assertEqual(1 / 20, darts.p(region=5))
        self.assertAlmostEqual(19 / 20, darts.p(region__ne=20))
        self.assertEqual(1 / 20, darts.p(region=5, region__ne=20))
        self.assertAlmostEqual(1 / 19, darts.given(region__ne=20).p(region=5))

    def test__1_1_2(self):

        population_counts = Series({
            'England': 60_776_238,
            'Scotland': 5_116_900,
            'Wales': 2_980_700
        })
        population_counts.index.name = 'country'
        populations = Discrete.from_counts(data=population_counts)
        self.assertAlmostEqual(0.882, populations.p(country='England'), 3)
        self.assertAlmostEqual(0.074, populations.p(country='Scotland'), 3)
        self.assertAlmostEqual(0.043, populations.p(country='Wales'), 3)
        language_probs = DataFrame.from_dict({
            'England': {'English': 0.95, 'Scottish': 0.04, 'Welsh': 0.01},
            'Scotland': {'English': 0.7, 'Scottish': 0.3, 'Welsh': 0.0},
            'Wales': {'English': 0.6, 'Scottish': 0.0, 'Welsh': 0.4}
        }, orient='columns')
        language_probs.index.name = 'language'
        language_probs.columns.name = 'country'
        language__given__country = Conditional(data=language_probs)
        self.assertIsInstance(language__given__country.data, DataFrame)
        self.assertListEqual(
            ['language'],
            language__given__country.joint_variables
        )
        self.assertListEqual(
            ['country'],
            language__given__country.conditional_variables
        )
        country__language = language__given__country * populations
        for prob, country, language in [
            (0.838, 'England', 'English'),
            (0.035, 'England', 'Scottish'),
            (0.009, 'England', 'Welsh'),
            (0.052, 'Scotland', 'English'),
            (0.022, 'Scotland', 'Scottish'),
            (0.0, 'Scotland', 'Welsh'),
            (0.026, 'Wales', 'English'),
            (0.0, 'Wales', 'Scottish'),
            (0.017, 'Wales', 'Welsh'),
        ]:
            self.assertAlmostEqual(
                prob,
                country__language.p(country=country, language=language),
                3
            )

    def test__example_1_2(self):

        has_kj = Discrete.from_probs(
            data={'yes': 1e-5, 'no': 1 - 1e-5},
            variables='has_kj'
        )
        self.assertEqual(1e-5, has_kj.p(has_kj='yes'))
        self.assertEqual(1 - 1e-5, has_kj.p(has_kj='no'))
        eats_hbs__given__has_kj = Conditional.from_probs(
            data={
                ('yes', 'yes'): 0.9,
                ('no', 'yes'): 0.1
            },
            joint_variables='eats_hbs',
            conditional_variables='has_kj'
        )
        eats_hbs = Discrete.from_probs(
            data={'yes': 0.5, 'no': 0.5},
            variables='eats_hbs'
        )
        # 1
        has_kj__given__eats_hbs = eats_hbs__given__has_kj * has_kj / eats_hbs
        self.assertEqual(
            1.8e-5,
            has_kj__given__eats_hbs.p(has_kj='yes', eats_hbs='yes')
        )
        # 2
        eats_hbs = Discrete.from_probs(
            data={'yes': 0.001, 'no': 0.999},
            variables='eats_hbs'
        )
        has_kj__given__eats_hbs = eats_hbs__given__has_kj * has_kj / eats_hbs
        self.assertEqual(
            9 / 1000,
            has_kj__given__eats_hbs.p(has_kj='yes', eats_hbs='yes')
        )

    def test__example_1_3(self):

        butler = Discrete.from_probs(
            data={'yes': 0.6, 'no': 0.4}, variables='butler'
        )
        maid = Discrete.from_probs(
            data={'yes': 0.2, 'no': 0.8}, variables='maid'
        )
        butler__and__maid = butler * maid
        knife__given__butler__and__maid = Conditional.from_probs(data={
                ('yes', 'no', 'no'): 0.3,
                ('yes', 'no', 'yes'): 0.2,
                ('yes', 'yes', 'no'): 0.6,
                ('yes', 'yes', 'yes'): 0.1,
                ('no', 'no', 'no'): 0.7,
                ('no', 'no', 'yes'): 0.8,
                ('no', 'yes', 'no'): 0.4,
                ('no', 'yes', 'yes'): 0.9,
            },
            joint_variables='knife_used',
            conditional_variables=['butler', 'maid']
        )
        butler__and__maid__and__knife = (
            knife__given__butler__and__maid * butler__and__maid
        )
        butler__given__knife = butler__and__maid__and__knife.given(
            knife_used='yes').p(butler='yes')
        self.assertAlmostEqual(0.728, butler__given__knife, 3)

    def test__example_1_4(self):

        occupied__given__alice__and__bob = Conditional.from_probs({
                (True, False, False): 1,
                (True, False, True): 1,
                (True, True, False): 1,
                (True, True, True): 0,
            },
            joint_variables='occupied',
            conditional_variables=['alice', 'bob']
        )
        alice__and__bob = Discrete.from_probs({
            (False, False): 0.25,
            (False, True): 0.25,
            (True, False): 0.25,
            (True, True): 0.25,
        }, variables=['alice', 'bob'])
        alice__and__bob__and__occupied = (
            occupied__given__alice__and__bob * alice__and__bob
        )
        self.assertEqual(
            1,
            alice__and__bob__and__occupied.given(
                alice=True, occupied=True
            ).p(bob=False)
        )

    def test__example_1_7(self):

        c__given__a__and__b = Conditional.binary_from_probs(
            data={
                (0, 0): 0.1,
                (0, 1): 0.99,
                (1, 0): 0.8,
                (1, 1): 0.25,
            },
            joint_variable='C',
            conditional_variables=['A', 'B']
        )
        a = Discrete.binary(0.65, 'A')
        b = Discrete.binary(0.77, 'B')
        a__and__b = a * b
        a__and__b__and__c = a__and__b * c__given__a__and__b
        self.assertAlmostEqual(
            0.8436,
            a__and__b__and__c.given(C=0).p(A=1),
            4
        )

    def test__1_3_1(self):

        t = Discrete.from_observations(
            data=DataFrame({
                't': [s_a + s_b
                      for s_a, s_b in product(range(1, 7), range(1, 7))]
            })
        )
        s_a__s_b = Discrete.from_probs(
            data={
                (a, b): 1 / 36
                for a, b in product(range(1, 7), range(1, 7))
            },
            variables=['s_a', 's_b']
        )
        t_9__given__s_a__s_b = Conditional.from_probs(
            data={
                (9, a, b): int(a + b == 9)
                for a, b in product(range(1, 7), range(1, 7))
            },
            joint_variables=['t'],
            conditional_variables=['s_a', 's_b']
        )
        t_9__s_a__s_b = t_9__given__s_a__s_b * s_a__s_b
        t_9 = t_9__s_a__s_b / t.p(t=9)
        for s_a, s_b in product(range(1, 6), range(1, 6)):
            self.assertEqual(
                t_9.p(s_a=s_a, s_b=s_b),
                0.25 if s_a + s_b == 9 else 0
            )
