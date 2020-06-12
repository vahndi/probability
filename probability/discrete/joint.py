from collections import OrderedDict
from itertools import product

from pandas import Series, DataFrame
from typing import List

from pgmpy.factors.discrete import JointProbabilityDistribution as PGMJPD


class Joint(object):

    @staticmethod
    def fill_and_sort_for_jpt(data: Series) -> Series:
        """
        Fill in missing values in a Series and sort so that
        it can be used for a JointProbabilityDistribution.
        """
        var_names = list(data.index.names)
        index_data = data.index.to_frame()
        index_uniques = OrderedDict([
            (col, sorted(index_data[col].unique()))
            for col in index_data.columns
        ])

        values = []
        for indexer in product(*list(index_uniques.values())):
            value = {
                var_name: var_value
                for var_name, var_value in zip(var_names, indexer)
            }
            if indexer in data.keys():
                value['p'] = data[indexer]
            else:
                value['p'] = 0
            values.append(value)

        new_data = DataFrame(values).sort_values(
            var_names[::-1]
        ).set_index(var_names)['p']

        return new_data

    def __init__(self, data: Series):
        """
        :param data: Series with an index column for each variable,
                     and values of probability of each index row.
        """
        self._data: Series = Joint.fill_and_sort_for_jpt(data)
        self._var_names: List[str] = [name for name in list(data.index.names)]

    @property
    def data(self) -> Series:
        return self._data
