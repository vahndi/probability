from pandas import Index, MultiIndex, Series, DataFrame
from typing import Any, Dict, List, Optional, Union

from probability.pandas.prob_utils import margin, condition, multiply, cond_name_and_symbol, given, valid_name_comparator, \
    cond_name


class DiscreteDistribution(object):

    def __init__(self, data: Series,
                 given_var_names: List[str] = None,
                 given_conditions: Optional[Dict[str, Any]] = None):
        """
        Create a new DiscreteDistribution e.g. `P(A,B,C,D)`.

        :param data: Series with an index column for each variable, and values of probability of each index row.
        :param given_var_names: List of names of conditioned variables with given values.
        :param given_conditions: Dict[{name}__{comparator}, value] for each conditioned variable.
        """
        self._data: Series = data.copy()
        self._joints: List[str] = list(data.index.names)
        self._given_vars = given_var_names or []
        self._given_conditions: Dict[str, Any] = given_conditions or {}
        self._var_names: List[str] = self._joints + self._given_vars

    # region constructors

    @staticmethod
    def from_dict(data: Dict[Union[str, int, tuple], float],
                  var_names: Union[str, List[str]],
                  **given_conditions) -> 'DiscreteDistribution':
        """
        Create a new joint distribution from a dictionary of probabilities or counts.

        :param data: Dictionary mapping values of random variables to their probabilities.
        :param var_names: Name of each random variable.
        :param given_conditions: Dict[{name}__{comparator}, value] for each conditioned variable.
        """
        first_key = list(data.keys())[0]
        if isinstance(first_key, tuple):
            index = MultiIndex.from_tuples(list(data.keys()),
                                           names=[var_names] if isinstance(var_names, str) else var_names)
        elif type(first_key) in (str, int):
            index = Index(list(data.keys()),
                          name=var_names[0] if isinstance(var_names, list) else var_names)
        else:
            raise TypeError('probs must be Dict[[Union[str, int, tuple], float]')
        data = Series(data=list(data.values()), index=index, name='p')
        return DiscreteDistribution(data,
                                    given_var_names=list(given_conditions.keys()),
                                    given_conditions=given_conditions)

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
        return DiscreteDistribution.from_dict(normalized, var_names=names)

    @staticmethod
    def from_observations(data: DataFrame) -> 'DiscreteDistribution':
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
        """
        Return a list of the names of all joint and conditioned variables.
        """
        return self._var_names

    @property
    def joints(self) -> List[str]:
        """
        Return a list of the names of all joint variables.
        """
        return self._joints

    @property
    def given_conditions(self) -> Dict[str, Any]:
        """
        Return the names and given values of conditioned variables.
        """
        return self._given_conditions

    @property
    def data(self) -> Series:
        """
        Return the underlying data for the distribution.
        """
        return self._data

    @property
    def name(self) -> str:
        """
        Return a name for the distribution based on the names of the joint variables and the names,
        conditional comparators and values of the given variables.
        """
        str_joints = ','.join(self._joints)
        strs_conditions = []
        for cond_var, cond_val in self.given_conditions.items():
            strs_conditions.append(cond_name_and_symbol(cond_var, cond_val, self.var_names))
        str_conditions = '|' + ','.join(strs_conditions) if strs_conditions else ''
        return f'P({str_joints}{str_conditions})'

    # endregion

    # region methods

    def margin(self, *margins) -> 'DiscreteDistribution':
        """
        Marginalize over variables not in `margins`.

        :param margins: List of variables to keep in the marginal distribution.
        """
        # check input variables
        if not len(margins) > 0:
            raise ValueError('Must pass at least one margin variable.')
        if not set(margins).issubset(self._joints):
            raise ValueError('All margins must be joint variables.')
        if not len(margins) < len(self._joints):
            raise ValueError('Must pass fewer than the total number of joint variables as margins.')
        # calculate marginal distribution
        data = margin(self._data, *margins)
        return DiscreteDistribution(data=data)

    def condition(self, *cond_var_names) -> 'ConditionalTable':
        """
        Condition on variables.

        :param cond_var_names: Names of variables to condition on.
        """
        # check input variables
        if not set(cond_var_names).issubset(self._joints):
            raise ValueError('All conditioning variables must be joint variables.')
        if not len(cond_var_names) < len(self._joints):
            raise ValueError('Cannot condition on all joint variables simultaneously.')
        # calculate conditional table
        data = condition(self._data, *cond_var_names)
        from probability.pandas.conditional_table import ConditionalTable
        return ConditionalTable(data, cond_variables=[cv for cv in cond_var_names])

    def given(self, **given_conditions) -> 'DiscreteDistribution':
        """
        Condition on values of variables.

        :param given_conditions: Dict[{name}__{comparator}, value] for each conditioned variable.
        """
        # check input variables
        names_comps = set(given_conditions.keys())
        if not all([valid_name_comparator(name_comp, self._joints) for name_comp in names_comps]):
            raise ValueError('Given variables must be members of joint distribution.')
        # calculate conditional distribution
        data = given(self._data, **given_conditions)
        cond_values = {**self._given_conditions, **given_conditions}
        return DiscreteDistribution(
            data=data,
            given_var_names=self._given_vars + [cond_name(k, self.var_names) for k in names_comps],
            given_conditions=cond_values
        )

    def p(self, **joint_vals) -> float:
        """
        Return the value of the probability distribution at the values of the joint variables.

        :param joint_vals: Dict[name, value] for each joint variable to find probability at.
        """
        # check input variables
        if not set(joint_vals.keys()) == set(self._joints):
            raise ValueError('Must specify a value for each conditioned variable.')
        # find probability of joint variable values
        var_vals = [joint_vals[name] for name in self._data.index.names]
        if len(var_vals) == 1:
            var_vals = var_vals[0]
        try:
            return self._data.xs(var_vals)
        except KeyError:
            # not all values are in distribution variable values
            return 0.0

    # endregion

    # region operators

    def __mul__(self, other: 'ConditionalTable') -> 'DiscreteDistribution':
        """
        Multiply the joint distribution e.g. `P(B)` by another ConditionalTable conditioned over this distribution's
        joint variables e.g. `P(A|B)` and return a new joint distribution i.e. `P(A)` over the ConditionalTable's
        joint variables i.e. `A`.

        :param other: Conditional Table with conditional variables matching this distributions joint variables.
        """
        if self.joints == other.cond_vars and not self.given_conditions:
            data = multiply(conditional=other.data, marginal=self.data)
            return DiscreteDistribution(data=data)
        else:
            raise ValueError('Cannot multiply these distributions.')

    def __rmul__(self, other: 'ConditionalTable') -> 'DiscreteDistribution':
        return self * other

    def __truediv__(self, other: 'DiscreteDistribution') -> 'ConditionalTable':

        if not set(other.joints).issubset(self.joints):
            raise ValueError(
                'All joint variables in denominator distribution must be present in numerator distribution to divide.'
            )
        if not len(other.joints) < len(self.joints):
            raise ValueError('Denominator distribution must contain fewer joint variables than numerator distribution.')
        return self.condition(*other.joints)

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
