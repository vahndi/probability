from pandas import Index, MultiIndex, Series
from typing import Any, Dict, List, Optional, Union

from probability.pandas.prob_utils import margin, condition, multiply


class DiscreteDistribution(object):

    _var_names: List[str]
    _joints: List[str]
    _givens: Dict[str, Any]
    _not_givens: List[str]

    def __init__(self, data: Series,
                 givens: Optional[Dict[str, Any]] = None,
                 not_givens: Optional[List[str]] = None):
        """
        Create a new DiscreteDistribution.
        :param data: Series with an index column for each variable, and values of probability of each index row.
        :param givens: Dictionary of conditional variables mapped to their value.
        :param not_givens: List of conditional variables without a given value.
        """
        self._data = data.copy()
        var_names = list(data.index.names)
        self._var_names = var_names
        self._givens = givens or {}
        self._not_givens = not_givens or []
        self._joints = [n for n in var_names
                        if n not in self._givens.keys()
                        and n not in self._not_givens]
        # do assertions here based on the conditionals and marginals
        # e.g. all p values for a given combination of values of givens and not-givens should sum to 1

    @staticmethod
    def from_dict(data: dict, names: Union[str, List[str]],
                  *not_givens, **givens) -> 'DiscreteDistribution':
        """
        Create a new joint distribution from a dictionary of probabilities.
        """
        if isinstance(list(data.keys())[0], str):
            index = Index(list(data.keys()),
                          name=names[0] if isinstance(names, list) else names)
        elif isinstance(list(data.keys())[0], tuple):
            index = MultiIndex.from_tuples(list(data.keys()),
                                           names=[names] if isinstance(names, str) else names)
        else:
            raise TypeError('probs must be Dict[str, float] or Dict[Tuple[str], float]')
        data = Series(data=list(data.values()), index=index, name='p')
        return DiscreteDistribution(data, not_givens=list(not_givens), givens=givens)

    @property
    def var_names(self) -> List[str]:
        return self._var_names

    @property
    def joints(self) -> List[str]:
        return self._joints

    @property
    def givens(self) -> List[str]:
        return list(self._givens.keys())

    @property
    def not_givens(self) -> List[str]:
        return self._not_givens

    @property
    def data(self) -> Series:
        return self._data

    @property
    def name(self):

        str_joints = ','.join(self._joints)
        strs_conditions = []
        for var_name in self._var_names:
            if var_name in self._givens.keys():
                strs_conditions.append(f'{var_name}={self._givens[var_name]}')
            elif var_name in self._not_givens:
                strs_conditions.append(var_name)
        str_conditions = '|' + ','.join(strs_conditions) if strs_conditions else ''
        return f'P({str_joints}{str_conditions})'

    def margin(self, *margins) -> 'DiscreteDistribution':
        """
        Marginalize over variables not in margins.

        :param margins: List of variables to keep.
        """
        data = margin(self._data, *margins)
        return DiscreteDistribution(
            data=data, givens=self._givens, not_givens=self._not_givens
        )

    def condition(self, *not_givens, **givens) -> 'DiscreteDistribution':
        """
        Condition on not-given and given variables.
        """
        given_overlap = set(givens.keys()).intersection(self._givens.keys())
        assert len(given_overlap) == 0, f'Cannot recondition values of {given_overlap}'
        not_givens = [g for g in self._givens
                      if g not in not_givens] + list(not_givens)  # need to recondition on existing not-givens
        data = condition(self._data, *not_givens, **givens)
        new_givens = {**self._givens, **givens}
        return DiscreteDistribution(
            data=data, givens=new_givens, not_givens=not_givens
        )

    def __mul__(self, other: 'DiscreteDistribution'):

        if (
                self.not_givens == other.joints and
                not self.givens and
                not other.not_givens and
                not other.givens
        ):
            # P(a) = P(a|b) * P(b)
            data = multiply(conditional=self.data, marginal=other.data)
            return DiscreteDistribution(
                data=data
            )
        elif (
                other.not_givens == self.joints and
                not other.givens and
                not self.not_givens and
                not self.givens
        ):
            # P(b) = P(b|a) * P(a)
            return other * self
        else:
            raise ValueError('Cannot multiply these distributions.')

    def __truediv__(self, other):

        # TODO: implement division of distributions with identical joints
        #  (and no conditionals or marginalizeds initially) to finish Bayes rule
        raise NotImplementedError

    def __repr__(self):
        return self.name

    def __str__(self):
        return str(self.data)


prior = DiscreteDistribution.from_dict({'blue': 0.6, 'red': 0.4}, 'box')

cond_data = {
    ('apple', 'blue'):  0.75,
    ('orange', 'blue'): 0.25,
    ('apple', 'red'):  0.25,
    ('orange', 'red'): 0.75
}
cond = DiscreteDistribution.from_dict(cond_data, ['fruit', 'box'], 'box')