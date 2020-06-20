from collections import OrderedDict
from itertools import product
from pandas import Series, DataFrame
from typing import List, Dict, Union

from pgmpy.factors.discrete import JointProbabilityDistribution as JPD


class Joint(object):

    # region constructors

    @staticmethod
    def fill_and_sort_for_jpt(data: Series) -> Series:
        """
        Fill in missing values in a Series and sort so that
        it can be used for a JointProbabilityDistribution.
        """
        variable_names = list(data.index.names)
        index_data = data.index.to_frame()
        variable_states = OrderedDict([
            (col, sorted(index_data[col].unique()))
            for col in index_data.columns
        ])

        values = []
        for indexer in product(*list(variable_states.values())):
            value = {
                var_name: var_value
                for var_name, var_value in zip(variable_names, indexer)
            }
            if len(indexer) == 1:
                indexer = indexer[0]
            if indexer in data.keys():
                value['p'] = data[indexer]
            else:
                value['p'] = 0
            values.append(value)

        new_data = DataFrame(values).sort_values(
            variable_names[::-1]
        ).set_index(variable_names)['p']

        return new_data

    def __init__(self, jpd: JPD):
        """
        Wrapper for a pgmpy JointProbabilityDistribution.

        https://pgmpy.org/_modules/pgmpy/factors/discrete/JointProbabilityDistribution.html
        https://pgmpy.org/_modules/pgmpy/factors/discrete/DiscreteFactor.html

        :param jpd: pgmpy JointProbabilityDistribution
        """
        self._jpd: JPD = jpd

    @staticmethod
    def from_series(data: Series) -> 'Joint':
        """
        Create a new joint distribution from a series of probabilities.

        :param data: Series with an index column for each variable,
                     and values of probability of each index row.
        """
        # calculate JPD kwargs
        data: Series = Joint.fill_and_sort_for_jpt(data)
        variables: List[str] = [name for name in list(data.index.names)]
        cardinality: List[int] = data.index.to_frame().nunique().to_list()
        values = data.to_list()
        index_data = data.index.to_frame()
        state_names = {
            col: sorted(index_data[col].unique())
            for col in index_data.columns
        }
        # create JPD
        jpd = JPD(
            variables=variables,
            cardinality=cardinality,
            values=values
        )
        jpd.state_names = state_names
        return Joint(jpd)

    @staticmethod
    def from_observations(data: DataFrame) -> 'Joint':
        """
        Create a new joint distribution based on the counts of items in the
        given data.

        :param data: DataFrame where each column represents a discrete random
                     variable, and each row represents an observation.
        """
        prob_data: Series = (
            data.groupby(list(data.columns)).size() / len(data)
        ).rename('p')
        return Joint.from_series(prob_data)

    @staticmethod
    def from_dict(data: Dict[Union[str, int, tuple], float],
                  variables: Union[str, List[str]]) -> 'Joint':
        """
        Create a new joint distribution from a dictionary of probabilities or
        counts.

        :param data: Dictionary states of random variables to probabilities.
        :param variables: Name of each random variable.
        """
        if len(variables) == 1:
            prob_data = Series(data=data, name='p')
            prob_data.index.name = variables[0]
            return Joint.from_series(data=prob_data)
        else:
            prob_data: Series = Series(data=data, name='p')
            prob_data.index.names = variables
            return Joint.from_series(data=prob_data)

    @staticmethod
    def from_counts(counts: Dict[Union[str, int, tuple], int],
                    variables: Union[str, List[str]]) -> 'Joint':
        """
        Return a new joint distribution from counts of random variable values.

        :param counts: Dictionary mapping values of random variables to the
                       number of observations.
        :param variables: Name of each random variable.
        """
        raise NotImplementedError

    # endregion

    @property
    def data(self) -> Series:
        """
        Return the Joint data Series.
        """
        return Series(data=self._jpd.values)

    @property
    def variables(self) -> List[str]:
        """
        Return a list of variable names in the joint distribution.
        """
        return self._jpd.variables

    @property
    def jpd(self) -> JPD:
        """
        Return the wrapped pgmpy JointProbabilityDistribution.
        """
        return self._jpd

    def __getitem__(self, item):
        """
        Return the distribution marginalized to the given item(s).

        :param item: The name(s) of the items to marginalize to.
        """
        raise NotImplementedError

    def __len__(self):

        return self._jpd.values.size
