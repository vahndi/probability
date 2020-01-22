from pandas import concat, DataFrame, Series
import re
from typing import Any, List, Optional


class VariableFilter(object):

    re_var_comparator = re.compile(
        r'(\w+)__((?:eq)|(?:ne)|(?:lt)|(?:gt)|(?:le)|(?:ge)|(?:in)|(?:not_in))'
    )
    var_name: str
    comparator: Optional[str]
    value: Any
    conditional: bool
    comparator_symbols = {
        'eq': lambda arg, val: f'{arg}={val}',
        'ne': lambda arg, val: f'{arg}≠{val}',
        'lt': lambda arg, val: f'{arg}<{val}',
        'gt': lambda arg, val: f'{arg}>{val}',
        'le': lambda arg, val: f'{arg}≤{val}',
        'ge': lambda arg, val: f'{arg}≥{val}',
        'in': lambda arg, vals: '{}∈{}'.format(
            arg, '{' + ",".join([str(val) for val in vals]) + '}'
        ),
        'not_in': lambda arg, vals: '{}∉{}'.format(
            arg, '{' + ",".join([str(val) for val in vals]) + '}'
        )
    }

    def __init__(self, arg_name: str, var_names: List[str], arg_value: Any = None):

        if arg_value is None:
            # no conditioning value given
            if arg_name not in var_names:
                raise ValueError(f'{arg_name} is not a variable in the dataset.')
            self.var_name = arg_name
            self.comparator = None
            self.value = None
            self.conditional = False
        else:
            # conditioning value given
            match = self.re_var_comparator.match(arg_name)
            if not match:
                if arg_name in var_names:
                    var_name, comparator = arg_name, 'eq'
                elif arg_name in ['_' + n for n in var_names]:
                    var_name, comparator = arg_name[1:], 'eq'
                else:
                    raise ValueError(f'Could not match {arg_name}.')
            else:
                var_name, comparator = match.groups()
            # set var_name
            if arg_name in var_names:
                self.var_name = arg_name
                self.conditional = False
            elif arg_name in ['_' + n for n in var_names]:
                self.var_name = arg_name[1:]
                self.conditional = True
            self.var_name: str = var_name
            # set comparator
            self.comparator = comparator
            self.value = arg_value

    def get_name(self) -> str:

        return self.comparator_symbols[self.comparator](self.var_name, self.value)

    def get_filter(self, data: DataFrame) -> Series:

        if self.comparator is None:
            return Series(data=[True]*len(data), index=data.index, name=self.var_name)

        name = self.get_name()
        compare: Series = data[self.var_name]
        if self.comparator == 'eq':
            return Series(data=compare == self.value, name=name)
        elif self.comparator == 'ne':
            return Series(data=compare != self.value, name=name)
        elif self.comparator == 'lt':
            return Series(data=compare < self.value, name=name)
        elif self.comparator == 'gt':
            return Series(data=compare > self.value, name=name)
        elif self.comparator == 'le':
            return Series(data=compare <= self.value, name=name)
        elif self.comparator == 'ge':
            return Series(data=compare >= self.value, name=name)
        elif self.comparator == 'in':
            return Series(data=compare.isin(self.value), name=name)
        elif self.comparator == 'not_in':
            return Series(data=~compare.isin(self.value), name=name)


class JointDistribution(object):

    def __init__(self, data: DataFrame):

        var_names = list(data.columns)
        # data validation
        if len(var_names) != len(set(var_names)):
            raise ValueError('Data column names must be unique.')
        non_identifiers = [var_name for var_name in var_names if not var_name.isidentifier()]
        if non_identifiers:
            raise ValueError('The following names cannot be used as variables: {}.'.format(non_identifiers))
        potential_errors = [var_name for var_name in var_names if var_name.startswith('_')]
        if potential_errors:
            raise ValueError('The following names could cause errors on conditioning: {}.'.format(potential_errors))
        # set member variables
        self._data: DataFrame = data
        self._var_names: List[str] = var_names
        for variable in self._var_names:
            setattr(self, variable, self._data[variable])

    def p(self, *args, **kwargs) -> Series:

        data = self._data
        # get filters
        filters: List[VariableFilter] = [
            VariableFilter(arg_name=arg, var_names=self._var_names)
            for arg in args
        ] + [
            VariableFilter(arg_name=kw_arg, var_names=self._var_names, arg_value=kw_value)
            for kw_arg, kw_value in kwargs.items()
        ]
        # conditionals e.g. c1 in p(j1, j2=a|c1=b)
        conditionals: List[VariableFilter] = [f for f in filters if f.conditional]
        if conditionals:
            condition_filters: DataFrame = concat(
                [f.get_filter(self._data) for f in conditionals], axis=1
            )
            conditions_filter: Series = condition_filters.all(axis=1)
            conditioned_data = data.loc[conditions_filter].copy()
        else:
            conditioned_data = data.copy()
        # marginals e.g. j2 in p(j1, j2=a|c1=b)
        marginals: List[VariableFilter] = [f for f in filters if not f.conditional and f.value is not None]
        marginal_names = [m.var_name for m in marginals]
        if marginals:
            marginal_filters: DataFrame = concat(
                [f.get_filter(conditioned_data) for f in marginals], axis=1
            )
            marginals_filter: Series = marginal_filters.all(axis=1)
            # replace data in each marginal with its marginal comparator and value (to make result explicit)
            for marginal in marginals:
                conditioned_data.loc[:, marginal.var_name] = marginal.get_name()
            marginal_data = conditioned_data.loc[marginals_filter]
        else:
            marginal_data = conditioned_data
        # fulls e.g. j1 in p(j1, j2=a|c1=b)
        fulls: List[VariableFilter] = [f for f in filters if not f.conditional and f.value is None]
        full_names = [f.var_name for f in fulls]
        final_data = marginal_data.groupby(full_names + marginal_names).size() / len(conditioned_data)
        return final_data
