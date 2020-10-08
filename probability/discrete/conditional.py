from typing import List, Dict, Optional, Union

from pandas import DataFrame, Series


class Conditional(object):

    def __init__(self,
                 data: DataFrame,
                 joint_variables: Optional[Union[str, List[str]]] = None,
                 conditional_variables: Optional[Union[str, List[str]]] = None,
                 states: Optional[Dict[str, list]] = None):

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
        self._states: Dict[str, list] = states

    @staticmethod
    def from_probs(
            data: Union[dict, Series],
            joint_variables: Union[str, List[str]],
            conditional_variables: Union[str, List[str]],
            states: Optional[Dict[str, list]] = None
    ) -> 'Conditional':

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

    @property
    def data(self) -> DataFrame:

        return self._data

    @property
    def joint_variables(self) -> List[str]:

        return self._joint_variables

    @property
    def conditional_variables(self) -> List[str]:

        return self._conditional_variables

    @property
    def variables(self) -> List[str]:

        return self._joint_variables + self._conditional_variables

    @property
    def states(self) -> Dict[str, list]:

        return self._states
