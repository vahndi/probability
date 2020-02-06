from pandas import DataFrame, Series
import re
from typing import Any, List, Optional

from probability.pandas.prob_utils import comparator_symbols


class VariableFilter(object):

    re_var_comparator = re.compile(
        r'(\w+)__((?:eq)|(?:ne)|(?:lt)|(?:gt)|(?:le)|(?:ge)|(?:in)|(?:not_in))'
    )
    var_name: str
    comparator: Optional[str]
    value: Any
    conditional: bool

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

        return comparator_symbols[self.comparator](self.var_name, self.value)

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
