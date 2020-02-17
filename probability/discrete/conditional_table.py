from pandas import Series
from typing import Dict, List, Union

from probability.discrete.discrete_distribution import DiscreteDistribution
from probability.discrete.prob_utils import margin, given


class ConditionalTable(object):

    def __init__(self, data: Series, cond_var_names: Union[str, List[str]]):
        """
        Create a new Conditional Table from a Series of probabilities.

        :param data: Series mapping values of random variables to their probabilities.
        :param cond_var_names: Names of conditional variables.
        """
        # self._data: Series = data.copy()
        self._var_names: List[str] = list(data.index.names)
        self._cond_var_names: List[str] = [cond_var_names] if isinstance(cond_var_names, str) else cond_var_names
        self._joints: List[str] = [n for n in self._var_names if n not in self._cond_var_names]
        var_order = self._joints + self._cond_var_names
        self._data = data.copy().reset_index()[var_order + ['p']].set_index(var_order)['p']

    # region attributes

    @staticmethod
    def from_dict(data: Dict[Union[str, int, tuple], float],
                  var_names: Union[str, List[str]],
                  cond_var_names: Union[str, List[str]] = None):
        """
        Create a new Conditional Table from a dictionary of probabilities.

        :param data: Dictionary mapping values of random variables to their probabilities.
        :param var_names: List of variable names for the elements of the
        :param cond_var_names: Names of conditional variables.
        """
        s_data = Series(data=data, name='p')
        s_data.index.names = [var_names] if isinstance(var_names, str) else var_names
        return ConditionalTable(data=s_data, cond_var_names=cond_var_names)

    @property
    def name(self) -> str:

        str_joints = ','.join(self._joints)
        strs_conditions = []
        for cond_var in self._cond_var_names:
            strs_conditions.append(cond_var)
        str_conditions = '|' + ','.join(strs_conditions) if strs_conditions else ''
        return f'P({str_joints}{str_conditions})'

    @property
    def data(self) -> Series:
        return self._data

    @property
    def cond_vars(self) -> List[str]:
        return self._cond_var_names

    # endregion

    # region methods

    def margin(self, *margins) -> 'ConditionalTable':
        """
        Create a new Conditional Table, marginalized to `margins`.

        :param margins: Names of variables to keep.
        """
        # check input arguments
        if not len(margins) > 0:
            raise ValueError('Must pass at least one margin variable.')
        if not set(margins).issubset(self._joints):
            raise ValueError('All margins must be joint variables.')
        # calculate marginal distributions
        data = margin(self._data, *margins, *self._cond_var_names)
        return ConditionalTable(data=data, cond_var_names=self._cond_var_names)

    def p(self, **var_vals) -> float:
        """
        Return the probability of the the specified values of the joint and conditional variables.

        N.B. need to supply all values for conditional and joint variables.

        :param var_vals: Values for each conditional and joint variable.
        """
        # check input arguments
        var_val_keys = set(var_vals.keys())
        has_all_conds = set(self._cond_var_names).intersection(var_val_keys) == set(self._cond_var_names)
        has_all_joints = set(self._joints).intersection(var_val_keys) == set(self._joints)
        if not has_all_conds and has_all_joints:
            raise ValueError('Must specify a value for each conditional and joint variable.')
        if not len(self._cond_var_names) + len(self._joints) == len(var_vals):
            raise ValueError('Too many variable values passed.')
        # calculate probability
        joint_cond_vals = [var_vals[name] for name in self._data.index.names]
        if len(joint_cond_vals) == 1:
            joint_cond_vals = joint_cond_vals[0]
        return self._data.xs(joint_cond_vals)

    def given(self, **given_vals) -> DiscreteDistribution:
        """
        Return the Discrete Distribution at the given values of the conditional variables.
        N.B. will give incorrect results if the conditional table was created from incomplete distributions.

        :param given_vals: Values of conditional variables to create probability distribution from.
        """
        # check input arguments
        given_val_keys = set(given_vals.keys())
        has_all_conds = set(self._cond_var_names).intersection(given_val_keys) == set(self._cond_var_names)
        if not has_all_conds:
            raise ValueError('Must supply values for all conditioned variables to get to joint distribution.')
        # calculate probability
        data = given(self._data, **given_vals)
        return DiscreteDistribution(data, given_conditions=given_vals)

    # endregion

    def __repr__(self):

        return self.name

    def __str__(self):
        return (
            f'ConditionalTable: {self.name}\n' +
            '=' * (18 + len(self.name)) + '\n' +
            str(self.data) + '\n' +
            f'sum: {self.data.sum()}'
        )
