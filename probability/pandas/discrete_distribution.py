from pandas import Index, MultiIndex, Series, DataFrame
from typing import Any, Dict, List, Optional, Union

from probability.pandas.prob_utils import margin, condition, multiply, name_and_symbol, given


class DiscreteDistribution(object):

    def __init__(self, data: Series,
                 cond_variables: Optional[List[str]] = None,
                 cond_values: Optional[Dict[str, Any]] = None):
        """
        Create a new DiscreteDistribution.

        :param data: Series with an index column for each variable, and values of probability of each index row.
        :param cond_values: Dictionary of conditional variables mapped to their value.
        :param cond_variables: List of conditional variables without a given value.
        """
        self._data: Series = data.copy()
        self._var_names: List[str] = list(data.index.names)
        self._cond_vars: List[str] = cond_variables or []
        self._cond_vals: Dict[str, Any] = cond_values or {}
        self._joints: List[str] = [n for n in self._var_names
                                   if n not in self._cond_vals.keys() and n not in self._cond_vars]
        # do assertions here based on the conditionals and marginals
        # e.g. all p values for a given combination of values of co and not-cond_values should sum to 1

    # region constructors

    @staticmethod
    def from_dict(data: Dict[Union[str, int, tuple], float],
                  names: Union[str, List[str]],
                  *cond_vars, **cond_values) -> 'DiscreteDistribution':
        """
        Create a new joint distribution from a dictionary of probabilities or counts.

        :param data: Dictionary mapping values of random variables to their probabilities.
        :param names: Name of each random variable.
        :param cond_vars: Variables to condition on without giving a value.
        :param cond_values: Variables to condition on with given values.
        """
        first_key = list(data.keys())[0]
        if isinstance(first_key, tuple):
            index = MultiIndex.from_tuples(list(data.keys()),
                                           names=[names] if isinstance(names, str) else names)
        elif type(first_key) in (str, int):
            index = Index(list(data.keys()),
                          name=names[0] if isinstance(names, list) else names)
        else:
            raise TypeError('probs must be Dict[[Union[str, int, tuple], float]')
        data = Series(data=list(data.values()), index=index, name='p')
        return DiscreteDistribution(data, cond_variables=list(cond_vars), cond_values=cond_values)

    @staticmethod
    def from_counts(counts: Dict[Union[str, int], int],
                    names: Union[str, List[str]]) -> 'DiscreteDistribution':
        """
        Return a new joint distribution from counts of random variable values.

        :param counts: Dictionary mapping values of random variables to the number of observations.
        :param names: Name of each random variable.
        """
        sum_values = sum(counts.values())
        normalized = {k: v / sum_values for k, v in counts.items()}
        return DiscreteDistribution.from_dict(normalized, names=names)

    @staticmethod
    def from_dataset(data: DataFrame) -> 'DiscreteDistribution':
        """
        Create a new discrete distribution based on the counts of items in the given data.

        :param data: DataFrame where each column represents a discrete random variable,
                     and each row represents an observation.
        """
        prob_data: Series = (data.groupby(list(data.columns)).size() / len(data)).rename('p')
        return DiscreteDistribution(prob_data)

    # endregion

    # region attributes

    @property
    def var_names(self) -> List[str]:
        return self._var_names

    @property
    def joints(self) -> List[str]:
        return self._joints

    @property
    def cond_values(self) -> Dict[str, Any]:
        """
        Return the names and specified values of conditional variables with specified values.
        """
        return self._cond_vals

    @property
    def cond_vars(self) -> List[str]:
        """
        Return the names of conditional variables without specified values.
        """
        return self._cond_vars

    @property
    def data(self) -> Series:
        return self._data

    @property
    def name(self) -> str:

        str_joints = ','.join(self._joints)
        strs_conditions = []
        for cond_var in self.cond_vars:
            strs_conditions.append(cond_var)
        for cond_var, cond_val in self.cond_values.items():
            strs_conditions.append(name_and_symbol(cond_var, cond_val, self.var_names))
        str_conditions = '|' + ','.join(strs_conditions) if strs_conditions else ''
        return f'P({str_joints}{str_conditions})'

    # endregion

    # region methods

    def margin(self, *margins) -> 'DiscreteDistribution':
        """
        Marginalize over variables not in margins.

        :param margins: List of variables to keep.
        """
        # check input variables
        if not len(margins) > 0:
            raise ValueError('Must pass at least one margin variable.')
        if not set(margins).issubset(self._joints):
            raise ValueError('All margins must be joint variables.')
        # calculate marginal distribution
        data = margin(self._data, *margins)
        return DiscreteDistribution(data=data)

    def condition(self, *cond_vars) -> 'ConditionalTable':
        """
        Condition on variables.

        :param cond_vars: List of names of variables to condition on.
        """
        # check input variables
        if not set(cond_vars).issubset(self._joints):
            raise ValueError('All conditioning variables must be joint variables.')
        if not len(cond_vars) < len(self._joints):
            raise ValueError('Cannot condition on all joint variables simultaneously.')
        # calculate conditional table
        data = condition(self._data, *cond_vars)
        from probability.pandas.conditional_table import ConditionalTable
        return ConditionalTable(data, cond_variables=[cv for cv in cond_vars])

    def given(self, **given_vals) -> 'DiscreteDistribution':
        """
        Condition on values of variables.

        :param given_vals: Values of variables to condition on to create new probability distribution.
        """
        # check input variables
        given_val_keys = set(given_vals.keys())
        if not given_val_keys.issubset(self._joints):
            raise ValueError('Given variables must be members of joint distribution.')
        # calculate conditional distribution
        data = given(self._data, **given_vals)
        cond_values = {**self._cond_vals, **given_vals}
        return DiscreteDistribution(data=data, cond_values=cond_values)

    def p(self, **joint_vals) -> float:
        """
        Return the value of the probability distribution p the values of the joint variables.

        :param joint_vals: Key-Value pairs to match.
        """
        # check input variables
        assert(set(joint_vals.keys()) == set(self._joints)), \
            'Must specify a value for each conditioned variable.'
        # find probability of joint variable values
        var_vals = [joint_vals[name] for name in self._data.index.names]
        if len(var_vals) == 1:
            var_vals = var_vals[0]
        return self._data.xs(var_vals)

    # endregion

    # region operators

    def __mul__(self, other: 'ConditionalTable') -> 'DiscreteDistribution':

        if (
                self.joints == other.cond_vars and
                not self.cond_vars and not self.cond_values
        ):
            # P(b) = P(b|a) * P(a)
            data = multiply(conditional=other.data, marginal=self.data)
            return DiscreteDistribution(data=data)
        else:
            raise ValueError('Cannot multiply these distributions.')

    def __rmul__(self, other: 'ConditionalTable') -> 'DiscreteDistribution':

        return self * other

    def __truediv__(self, other):

        # TODO: implement division of distributions with identical joints
        #  (and no conditionals or marginalizeds initially) to finish Bayes rule
        raise NotImplementedError

    # endregion

    def __repr__(self):

        return self.name

    def __str__(self):

        return (
            f'DiscreteDistribution: {self.name}\n' +
            '=' * (22 + len(self.name)) + '\n' +
            str(self.data) + '\n' +
            f'sum: {self.data.sum()}'
        )
