from pandas import concat, DataFrame, Series
from typing import List, Tuple, Any


class Condition(object):

    def __init__(self, var_name: str, comparator: str, value: Any):

        self.var_name: str = var_name
        self.comparator: str = comparator
        self.value: Any = value


class JointDistribution(object):

    def __init__(self, data: DataFrame):

        self._data: DataFrame = data
        self._variables: List[str] = list(self._data.columns)
        self._conditions: List[Tuple[str, str, Any]] = []  # var_name, comparator, value
        self._marginals: List[str] = []

        for variable in self._variables:
            try:
                setattr(self, variable, self._data[variable])
            except:
                print('Warning - could not set variable {}'.format(variable))

    def p(self, *args, **kwargs) -> Series:

        data = self._data
        # get margins
        margins = [var_name for var_name in args]
        # get conditions
        conditions = [Condition(var_name, '=', value) for var_name, value in kwargs.items()]
        # apply conditions
        if conditions:
            condition_filters = []
            for condition in conditions:
                condition_filter = self._data[condition.var_name] == condition.value
                condition_filters.append(condition_filter)
            conditions_filter: Series = concat(condition_filters, axis=1).all(axis=1)
            data = data.loc[conditions_filter]
        # remove marginalized and conditional variables
        data = data.loc[:, margins]
        # sum over remaining variables
        data = data.groupby(margins).size()
        # normalize
        data = data / data.sum()
        # return result
        data.name = 'p({}{}{})'.format(
            ','.join(margins),
            '|' if conditions else '',
            ','.join([
                '{}{}{}'.format(condition.var_name, condition.comparator, condition.value)
                for condition in conditions
            ])
        )
        return data
