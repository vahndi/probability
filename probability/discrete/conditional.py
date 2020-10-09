from typing import List, Dict, Optional, Union, TYPE_CHECKING

from pandas import DataFrame, Series, MultiIndex

if TYPE_CHECKING:
    from probability.discrete import Discrete

from probability.discrete.mixins import StatesMixin


class Conditional(
    StatesMixin,
    object
):

    def __init__(
            self,
            data: DataFrame,
            joint_variables: Optional[Union[str, List[str]]] = None,
            conditional_variables: Optional[Union[str, List[str]]] = None,
            states: Optional[Dict[str, list]] = None
    ):

        """
        Create a new Conditional Probability Table.

        :param data: DataFrame with probability variables as index and
                     conditional variables as columns.
        :param joint_variables: Names for the probability variables, if the
                                index level(s) are not named.
                                Overwrites DataFrame names - do not use to
                                reorder variables.
        :param conditional_variables: Names for the conditional variables, if
                                      the column level(s) are not named.
                                      Overwrites DataFrame names - do not use to
                                      reorder variables.
        """
        self._data: DataFrame = data
        if joint_variables is not None:
            if isinstance(joint_variables, str):
                joint_variables = [joint_variables]
            self._data.index.names = joint_variables
        if conditional_variables is not None:
            if isinstance(conditional_variables, str):
                conditional_variables = [conditional_variables]
            self._data.columns.names = conditional_variables
        self._joint_variables = list(data.index.names)
        self._conditional_variables = list(data.columns.names)
        if states is None:
            states = {
                **{
                    variable: sorted(
                        self._data.index.get_level_values(variable).unique()
                    ) for variable in self._joint_variables
                },
                **{
                    variable: sorted(
                        self._data.columns.get_level_values(variable).unique()
                    ) for variable in self._conditional_variables
                }
            }
        else:
            if set(states.keys()) != set(joint_variables +
                                         conditional_variables):
                raise ValueError('states must match variables')
        self._states: Dict[str, list] = states

    @staticmethod
    def from_probs(
            data: Union[dict, Series],
            joint_variables: Union[str, List[str]],
            conditional_variables: Union[str, List[str]],
            states: Optional[Dict[str, list]] = None
    ) -> 'Conditional':
        """
        Create a conditional probability table from a Series of data
        probabilities.
        N.B. for a distribution with j joint variables and c conditional
        variables, index columns must be in the order:
            [joint_1, ..., joint_j, conditional_1, .... conditional_c]

        :param data: Series with joint and conditional variable states in the
                     index, and p(joints|conditionals) as values.
        :param joint_variables: Joint variable name or names.
        :param conditional_variables: Conditional variable name or names.
        :param states: Optional dictionary mapping variable names to their
                       possible states. If not given, uses states present in the
                       data.
        """
        if isinstance(data, dict):
            data = Series(data)
        if isinstance(joint_variables, str):
            joint_variables = [joint_variables]
        if isinstance(conditional_variables, str):
            conditional_variables = [conditional_variables]

        variables = joint_variables + conditional_variables
        if None in data.index.names:
            data.index.names = variables
        if states is None:
            states = {
                variable: sorted(
                    data.index.get_level_values(variable).unique()
                )
                for variable in variables
            }
        data = data.unstack(level=conditional_variables)
        return Conditional(
            data=data,
            joint_variables=joint_variables,
            conditional_variables=conditional_variables,
            states=states
        )

    @staticmethod
    def binary_from_probs(
            data: Union[dict, Series],
            joint_variable: str,
            conditional_variables: Union[str, List[str]],
            conditional_states: Optional[Dict[str, list]] = None
    ) -> 'Conditional':
        """
        Create a conditional probability table for a binary variable e.g. B
        using probabilities that p(B) = 1, given different values of the
        conditional variables.

        :param data: Series with conditional variable states in the index
                     columns and p(B|C1, ...) = 1 for the values.
        :param joint_variable: Name of the joint variable.
        :param conditional_variables: Name(s) of the conditional variables.
        :param conditional_states: Optional mapping of names of  conditional
                                   variables to their possible states. If not
                                   given, uses states present in the data.
        """
        if isinstance(data, dict):
            data = Series(data)
        joint_variables = [joint_variable]
        if isinstance(conditional_variables, str):
            conditional_variables = [conditional_variables]
        binary_data = {}
        for ix, value in data.items():
            if isinstance(data.index, MultiIndex):
                binary_data[tuple([1] + list(ix))] = value
                binary_data[tuple([0] + list(ix))] = 1 - value
            else:
                binary_data[(ix, 1)] = value
                binary_data[(ix, 0)] = 1 - value
        if conditional_states is not None:
            states = {
                **conditional_states,
                **{joint_variable: [0, 1]}
            }
        else:
            states = None
        return Conditional.from_probs(
            data=binary_data,
            joint_variables=joint_variables,
            conditional_variables=conditional_variables,
            states=states
        )

    @property
    def data(self) -> DataFrame:

        return self._data

    @property
    def joint_variables(self) -> List[str]:
        """
        Return a list of the names of the joint variables.
        """
        return self._joint_variables

    @property
    def conditional_variables(self) -> List[str]:
        """
        Return a list of the names of the conditional variables.
        """
        return self._conditional_variables

    @property
    def variables(self) -> List[str]:
        """
        Return a list of the names of all of the variables.
        """
        return self._joint_variables + self._conditional_variables

    def given(self, **given_conditions) -> Union['Conditional', 'Discrete']:
        """
        Condition on values of variables. If values are given for all
        conditional variables, returns a Discrete distribution, otherwise
        returns a Conditional with variables matching that subset of conditions.

        :param given_conditions: Dict[{name}__{comparator}, value] for each
                                 conditioned variable.
        """
        condition_names = given_conditions.keys()
        if not set(condition_names).issubset(self._conditional_variables):
            raise ValueError('given variables is not subset of conditions')
        elif set(condition_names) == set(self._conditional_variables):
            from probability.discrete import Discrete
            selector = [given_conditions[variable]
                        for variable in self._conditional_variables]
            discrete_data = self._data[tuple(selector)]
            return Discrete.from_probs(
                data=discrete_data,
                variables=self._joint_variables
            )
        else:
            given_vars = list(given_conditions.keys())
            selectors = [
                col for col in self._data.columns
                if all(
                    col[
                        self._conditional_variables.index(variable)
                    ] == given_conditions[variable]
                    for variable in given_vars
                )
            ]
            cond_data = self._data[selectors]
            drop_cols = given_vars if len(given_vars) > 1 else given_vars[0]
            cond_data = cond_data.droplevel(drop_cols, axis=1)
            cond_vars = [var for var in self._conditional_variables
                         if var not in given_vars]
            return Conditional(
                data=cond_data,
                joint_variables=self.joint_variables,
                conditional_variables=cond_vars
            )

    def __repr__(self):

        str_joints = ','.join(self._joint_variables)
        str_conds = ','.join(self._conditional_variables)
        return f'p({str_joints}|{str_conds})'
