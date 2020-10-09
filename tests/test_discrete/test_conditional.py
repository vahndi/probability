from unittest.case import TestCase

from pandas import DataFrame, Series

from probability.discrete import Conditional


class TestConditional(TestCase):

    def setUp(self) -> None:

        self.languages = ['English', 'Scottish', 'Welsh']
        self.countries = ['England', 'Scotland', 'Wales']
        self.states = {
            'language': self.languages,
            'country': self.countries
        }

    @staticmethod
    def get_language_probs() -> DataFrame:

        return DataFrame.from_dict({
            'England': {'English': 0.95, 'Scottish': 0.04, 'Welsh': 0.01},
            'Scotland': {'English': 0.7, 'Scottish': 0.3, 'Welsh': 0.0},
            'Wales': {'English': 0.6, 'Scottish': 0.0, 'Welsh': 0.4}
        }, orient='columns')

    def check_conditional(self, conditional: Conditional):

        self.assertIsInstance(conditional.data, DataFrame)
        self.assertEqual(['language'], conditional.joint_variables)
        self.assertEqual(['country'], conditional.conditional_variables)
        self.assertEqual(conditional.states, self.states)

    def test_init__with_vars(self):

        language_probs = self.get_language_probs()
        language__given__country = Conditional(
            data=language_probs,
            joint_variables='language',
            conditional_variables='country',
            states={
                'language': self.languages,
                'country': self.countries
            }
        )
        self.check_conditional(language__given__country)

    def test_init__with_names_and_vars(self):

        language_probs = self.get_language_probs()
        language_probs.index.name = 'language'
        language_probs.columns.name = 'country'
        language__given__country = Conditional(
            data=language_probs,
            joint_variables='language',
            conditional_variables='country',
            states={
                'language': self.languages,
                'country': self.countries
            }
        )
        self.check_conditional(language__given__country)

    def test_init__with_names(self):

        language_probs = self.get_language_probs()
        language_probs.index.name = 'language'
        language_probs.columns.name = 'country'
        language__given__country = Conditional(data=language_probs)
        self.check_conditional(language__given__country)

    def test_from_probs_with_dict(self):

        probs = {
            ('English', 'England'): 0.95,
            ('English', 'Scotland'): 0.7,
            ('English', 'Wales'): 0.6,
            ('Scottish', 'England'): 0.04,
            ('Scottish', 'Scotland'): 0.3,
            ('Scottish', 'Wales'): 0.0,
            ('Welsh', 'England'): 0.01,
            ('Welsh', 'Scotland'): 0.0,
            ('Welsh', 'Wales'): 0.4,
        }
        language__given__country = Conditional.from_probs(
            data=probs,
            joint_variables='language',
            conditional_variables='country'
        )
        self.check_conditional(language__given__country)

    def test_from_probs_with_series(self):

        probs = Series({
            ('England', 'English'): 0.95,
            ('Scotland', 'English'): 0.7,
            ('Wales', 'English'): 0.6,
            ('England', 'Scottish'): 0.04,
            ('Scotland', 'Scottish'): 0.3,
            ('Wales', 'Scottish'): 0.0,
            ('England', 'Welsh'): 0.01,
            ('Scotland', 'Welsh'): 0.0,
            ('Wales', 'Welsh'): 0.4,
        })
        probs.index.names = ['country', 'language']
        language__given__country = Conditional.from_probs(
            data=probs,
            joint_variables='language',
            conditional_variables='country'
        )
        self.check_conditional(language__given__country)
        print(language__given__country)

    def test_binary_from_probs(self):

        c__a_b__1 = Conditional.from_probs(
            data={
                (1, 0, 0): 0.1,
                (1, 0, 1): 0.99,
                (1, 1, 0): 0.8,
                (1, 1, 1): 0.25,
                (0, 0, 0): 1 - 0.1,
                (0, 0, 1): 1 - 0.99,
                (0, 1, 0): 1 - 0.8,
                (0, 1, 1): 1 - 0.25,
            },
            joint_variables='C',
            conditional_variables=['A', 'B']
        ).data
        c__a_b__2 = Conditional.binary_from_probs(
            data={
                (0, 0): 0.1,
                (0, 1): 0.99,
                (1, 0): 0.8,
                (1, 1): 0.25,
            },
            joint_variable='C',
            conditional_variables=['A', 'B']
        ).data
        self.assertTrue(c__a_b__1.equals(c__a_b__2))
