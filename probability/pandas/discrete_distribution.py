from typing import Optional, List, Union, overload

from pandas import Series, DataFrame, merge


class DiscreteDistribution(object):

    _joint_vars: List[str]
    _conditional_vars: List[str]

    def __init__(self, data: Series,
                 conditionals: Optional[List[str]] = None):

        self._data = data.copy()
        var_names = data.index.names
        conditionals = conditionals or []
        self._var_names = var_names
        self._joint_vars = [n for n in var_names if n not in conditionals]
        self._conditional_vars = conditionals
        # do assertions here based on the conditionals and marginals
        # e.g. all p values for a given value of the conditional should sum to 1

    @property
    def joint_vars(self) -> List[str]:
        return self._joint_vars

    @property
    def conditional_vars(self) -> List[str]:
        return self._conditional_vars

    def marginalize(self, *margins) -> 'DiscreteDistribution':
        """
        Marginalize over variables not in margins.

        :param margins: List of variables to keep.
        """
        margins = [m for m in margins]
        data = self._data.copy()
        p_name = data.name
        if not p_name:
            data.name = self.name
            p_name = self.name
        data = data.to_frame().reset_index()
        data = data.groupby(margins)[p_name].sum()
        return DiscreteDistribution(
            data=data,
            conditionals=self._conditional_vars
        )

    def condition(self, *conditionals) -> 'ConditionalDistributions':  # need another type??
        """
        Condition on `conditionals`.

        :param conditionals: Names of variables to calculate conditional distributions on.
        """
        # P(A,B|C,D) = P(A,B,C,D) / P(C,D)
        # conditional = joint(all) / joint(conditions)
        data = self._data.copy()
        conditionals = [c for c in conditionals]
        p_name = data.name
        sum_name = p_name + '_sum'
        joint = data.reset_index()
        sums = joint.groupby(conditionals).sum().reset_index().rename(columns={p_name: sum_name})
        merged = merge(left=joint, right=sums, on=conditionals)
        merged['p'] = merged[p_name] / merged[sum_name]
        return DiscreteDistribution(
            data=merged[self._var_names + ['p']].set_index(self._var_names)['p'],
            conditionals=conditionals
        )

    @overload
    def __mul__(self, other: 'DiscreteDistribution') -> float:
        pass

    @overload
    def __mul__(self, other: 'ConditionalDistributions') -> 'DiscreteDistribution':
        pass

    def __mul__(self, other: Union['ConditionalDistributions']):

        if self.conditional_vars == other.joint_vars and not other.conditional_vars:
            # P(a) = P(a|b) * P(b)
            cond_data = self.data.rename(self.name).to_frame().reset_index()
            prior_data = other.data.rename(other.name).to_frame().reset_index()
            merged = merge(left=cond_data, right=prior_data, on=self.conditional_vars)
            merged['p'] = merged[self.name] * merged[other.name]
            results = merged.groupby(self.joint_vars)['p'].sum()
            results.name = f'{self.name} * {other.name}'
            return DiscreteDistribution(results)
        elif self.joint_vars == other.conditional_vars and not self.conditional_vars:
            # P(b) = P(b|a) * P(a)
            cond_data = other.data.rename(other.name).to_frame().reset_index()
            prior_data = self.data.rename(self.name).to_frame().reset_index()
            merged = merge(left=cond_data, right=prior_data, on=other.conditional_vars)
            merged['p'] = merged[other.name] * merged[self.name]
            results = merged.groupby(other.joint_vars)['p'].sum()
            results.name = f'{other.name} * {self.name}'
            return DiscreteDistribution(results)

    def __truediv__(self, other):

        # TODO: implement division of distributions with identical joint_vars
        #  (and no conditionals or marginalizeds initially) to finish Bayes rule
        raise NotImplementedError

    @property
    def data(self) -> Series:
        return self._data

    @property
    def name(self):
        return 'P({}{}{})'.format(
            ','.join(self.joint_vars),
            '|' if self.conditional_vars else '',
            ','.join(self.conditional_vars)
        )

    def __repr__(self):
        return self.name

    def __str__(self):
        return str(self.data)


prior_data = DataFrame({'box': ['blue', 'red'], 'p': [0.6, 0.4]}).set_index('box')['p']
prior = DiscreteDistribution(data=prior_data)

cond_data = DataFrame({
    'fruit': ['apple', 'orange', 'apple', 'orange'],
    'box': ['blue', 'blue', 'red', 'red'],
    'p': [0.75, 0.25, 0.25, 0.75]
}).set_index(['fruit', 'box'])['p']
cond = DiscreteDistribution(data=cond_data, conditionals=['box'])

