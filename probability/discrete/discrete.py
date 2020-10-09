from itertools import product
from typing import Union, List, Dict, overload, Optional

from pandas import Series, DataFrame, MultiIndex, merge

from probability.discrete.conditional import Conditional
from probability.discrete.mixins import StatesMixin
from probability.discrete.prob_utils import p, given, valid_name_comparator


class Discrete(
    StatesMixin,
    object
):

    @overload
    def __init__(self, data: Series,
                 variables: str,
                 states: List[str]):
        pass

    @overload
    def __init__(self, data: Series,
                 variables: List[str],
                 states: Dict[str, List[str]]):
        pass

    def __init__(self, data: Series,
                 variables: Union[str, List[str]],
                 states: Union[list, Dict[str, list]]):

        self._data = data
        if isinstance(variables, str):
            variables = [variables]
        self._variables: List[str] = variables
        if isinstance(states, list):
            states = {self._variables[0]: states}
        self._states: Dict[str, list] = states
        self._data.name = f'p({",".join(self._variables)})'

    @property
    def variables(self) -> List[str]:

        return self._variables

    @staticmethod
    def from_counts(
            data: Union[Series, dict],
            variables: Optional[Union[str, List[str]]] = None,
            states: Optional[Union[list, Dict[str, list]]] = None
    ) -> 'Discrete':
        """
        Return a new joint distribution from counts of random variable values.

        :param data: Series mapping values of random variables to the
                       number of observations. Index column(s) should be named
                       after the variables.
        :param variables: Optional variable name or list of variable names.
                          Required if Series index columns are not named.
                          Overrides Series index column names.
        :param states: Optional list of states for single variable or dict of
                       variables to states for multiple variables. Use when
                       there are possible states not represented in the data.
        """
        if isinstance(data, dict):
            data = Series(data)

        # assign variables
        if not isinstance(data.index, MultiIndex):
            if variables is None:
                index_name = data.index.name
                if index_name is None:
                    raise ValueError(
                        'Must give variable name or name index column'
                    )
                else:
                    variables = [index_name]
            else:
                if isinstance(variables, str):
                    variables = [variables]
                elif not isinstance(variables, list):
                    raise ValueError('variables must be type str or List[str]')
        else:
            if variables is None:
                index_names = list(data.index.names)
                if None in index_names:
                    raise ValueError(
                        'Must give variable name or name index column'
                    )
                else:
                    variables = index_names
        data.index.names = variables

        # assign states
        if states is None:
            states = {
                variable: sorted(
                    data.index.get_level_values(variable).unique()
                ) for variable in variables
            }
        elif isinstance(states, list):
            if len(variables) != 1:
                raise ValueError(
                    'Must give dict of states if more than one variable'
                )
            else:
                states = {variables[0]: states}
        elif isinstance(states, dict):
            if not set(variables) == set(states.keys()):
                raise ValueError(
                    'state names do not match variable names'
                )

        # create distribution
        probs = data / data.sum()
        return Discrete(
            data=probs, variables=variables, states=states
        )

    @staticmethod
    def from_observations(
            data: DataFrame,
            variables: Optional[Union[str, List[str]]] = None,
            states: Optional[Union[list, Dict[str, list]]] = None
    ) -> 'Discrete':
        """
        Create a new joint distribution based on observations of items in the
        given data.

        :param data: DataFrame where each column represents a discrete random
                     variable, and each row represents an observation.
        :param variables: Optional variable name or list of variable names.
                          Overrides DataFrame column names.
                          Do not use to try to reorder variables.
        :param states: Optional list of states for single variable or dict of
                       variables to states for multiple variables. Use when
                       there are possible states not represented in the data.
        """
        # assign variables
        if variables is None:
            variables = data.columns.to_list()
        elif isinstance(variables, str):
            variables = [variables]
        elif not isinstance(variables, list):
            raise ValueError('variables must be None, str or List[str]')
        data.columns = variables

        # assign states
        if states is None:
            states = {
                variable: sorted(data[variable].unique())
                for variable in variables
            }
        elif isinstance(states, list):
            if len(variables) != 1:
                raise ValueError(
                    'Must give dict of states if more than one variable'
                )
            else:
                states = {variables[0]: states}
        elif isinstance(states, dict):
            if not set(variables) == set(states.keys()):
                raise ValueError(
                    'state names do not match variable names'
                )

        # create distribution
        prob_data: Series = (data.groupby(variables).size() / len(data))
        return Discrete(data=prob_data, variables=variables, states=states)

    @staticmethod
    def from_probs(
            data: Union[Series, dict],
            variables: Optional[Union[str, List[str]]] = None,
            states: Optional[Union[list, Dict[str, list]]] = None
    ) -> 'Discrete':
        """
        Return a new joint distribution from counts of random variable values.

        :param data: Series mapping values of random variables to the
                       number of observations. Index column(s) should be named
                       after the variables.
        :param variables: Optional variable name or list of variable names.
                          Required if Series index columns are not named.
                          Overrides Series index column names.
        :param states: Optional list of states for single variable or dict of
                       variables to states for multiple variables. Use when
                       there are possible states not represented in the data.
        """
        return Discrete.from_counts(
            data=data, variables=variables, states=states
        )

    @staticmethod
    def binary(prob: float, variable: str):
        """
        Create a single variable binary probability.

        :param prob: P(variable = 1)
        :param variable: Name of variable.
        """
        return Discrete.from_probs(
            {0: 1 - prob, 1: prob}, variables=variable
        )

    @property
    def data(self) -> Series:

        return self._data

    def p(self, **kwargs):

        return p(self._data, **kwargs)

    def given(self, **given_conditions) -> 'Discrete':
        """
        Condition on values of variables.

        :param given_conditions: Dict[{name}__{comparator}, value] for each
                                 conditioned variable.
        """
        # check input variables
        names_comps = set(given_conditions.keys())
        if not all([valid_name_comparator(name_comp, self._variables)
                    for name_comp in names_comps]):
            raise ValueError(
                'Given variables must be members of joint distribution.'
            )
        # calculate conditional distribution
        data = given(self._data, **given_conditions)
        variables = [var for var in self._variables
                     if var not in given_conditions.keys()]
        states = {
            variable: self._states[variable]
            for variable in variables
        }
        return Discrete(
            data=data, variables=variables, states=states
        )

    def conditional(self, *conditionals) -> Conditional:
        """
        Return a Conditional table  the distribution on the conditional
        variables.

        :param conditionals: Names of variables to condition over every value of
        """
        col_names = self._data.index.names
        joint_variables = [n for n in col_names if n not in conditionals]
        variables = [n for n in col_names if n not in conditionals]
        variables.extend([n for n in col_names if n in conditionals])
        data = self._data.copy().rename('p_cond').reset_index()
        conditionals = list(conditionals)
        if conditionals:
            # find total probabilities for each combination of unique values in
            # the conditional variables e.g. P(C)
            sums = data.groupby(conditionals).sum().reset_index()
            # normalize each individual probability e.g. p(Ai,Bj,Ck,Dl) to
            # probability of its conditional values p(Ck)
            sums = sums[conditionals + ['p_cond']].rename(
                columns={'p_cond': 'p_sum'})
            merged = merge(left=data, right=sums, on=conditionals)
            merged['p_cond'] = merged['p_cond'] / merged['p_sum']
            data = merged[variables + ['p_cond']]
        data = data.set_index(variables)['p_cond']
        return Conditional.from_probs(
            data=data,
            joint_variables=joint_variables,
            conditional_variables=conditionals
        )

    def __mul__(self, other: Union[Conditional, 'Discrete']) -> 'Discrete':
        """
        Multiply by another Discrete or by a Conditional.
        """
        if isinstance(other, Conditional):
            data = (other.data * self._data).unstack()
            return Discrete(
                data=data,
                variables=data.index.names,
                states=self._states
            )
        elif isinstance(other, Discrete):
            results = {}
            for (s1k, s1v), (s2k, s2v) in product(
                self._data.items(), other._data.items()
            ):
                if isinstance(s1k, tuple):
                    k1 = [x for x in s1k]
                else:
                    k1 = [s1k]
                if isinstance(s2k, tuple):
                    k2 = [x for x in s2k]
                else:
                    k2 = [s2k]
                key = tuple(k1 + k2)
                results[key] = s1v * s2v
            data = Series(results)
            variables = (
                list(self._data.index.names) +
                list(other._data.index.names)
            )
            data.index.names = variables
            return Discrete(
                data=data,
                variables=variables,
                states={**self._states, **other._states}
            )

    def __rmul__(self, other):

        return self * other

    def __truediv__(self, other: 'Discrete') -> 'Discrete':

        data = self.data / other.data
        return Discrete(
            data=data,
            variables=data.index.names,
            states=self._states
        )

    def __repr__(self):

        return f'p({",".join(self._variables)})'
