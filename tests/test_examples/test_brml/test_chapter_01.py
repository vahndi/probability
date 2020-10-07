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
        populations = Discrete.from_counts(counts=population_counts)
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
        language__given__country = Conditional(
            data=language_probs,
            variables=['language', 'country'],
            states={
                'language': ['English', 'Scottish', 'Welsh'],
                'country': ['England', 'Scotland', 'Wales']
            }
        )
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
