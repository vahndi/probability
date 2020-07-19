from pandas import Index, MultiIndex, Series, DataFrame
from typing import Any, Dict, List, Optional, Union, overload, TYPE_CHECKING


from probability.discrete.prob_utils import margin, condition, multiply, \
    cond_name_and_symbol, given, valid_name_comparator, cond_name, p

if TYPE_CHECKING:
    from probability.discrete import ConditionalTable


class DiscreteDistribution(object):

    def __init__(self, data: Series,
                 cond_var_names: List[str] = None,
                 given_conditions: Optional[Dict[str, Any]] = None):
        """
        Create a new DiscreteDistribution e.g. `P(A,B,C,D)`.

        :param data: Series with an index column for each variable,
                     and values of probability of each index row.
        :param cond_var_names: List of names of conditioned variables
                               without given values.
        :param given_conditions: Dict[{name}__{comparator}, value]
                                 for each conditioned variable.
        """
        self._data: Series = data.copy()
        self._cond_var_names = cond_var_names or []
        self._joints: List[str] = [
            name for name in list(data.index.names)
            if name not in self._cond_var_names
        ]
        self._given_conditions: Dict[str, Any] = given_conditions or {}
        self._var_names: List[str] = (
            self._joints +
            self._cond_var_names +
            [cond_name(k) for k in self._given_conditions.keys()]
        )

    # region constructors

    @staticmethod
    def from_dict(data: Dict[Union[str, int, tuple], float],
                  var_names: Union[str, List[str]],
                  **given_conditions) -> 'DiscreteDistribution':
        """
        Create a new joint distribution from a dictionary of probabilities or
        counts.

        :param data: Dictionary mapping values of random variables to their
                     probabilities.
        :param var_names: Name of each random variable.
        :param given_conditions: Dict[{name}__{comparator}, value] for each
                                 conditioned variable.
        """
        first_key = list(data.keys())[0]
        if isinstance(first_key, tuple):
            index = MultiIndex.from_tuples(
                tuples=list(data.keys()),
                names=[var_names] if isinstance(var_names, str) else var_names
            )
        elif type(first_key) in (str, int):
            index = Index(
                data=list(data.keys()),
                name=var_names[0] if isinstance(var_names, list) else var_names
            )
        else:
            raise TypeError(
                'probs must be Dict[[Union[str, int, tuple], float]'
            )
        data = Series(data=list(data.values()), index=index, name='p')
        return DiscreteDistribution(
            data=data,
            cond_var_names=list(given_conditions.keys()),
            given_conditions=given_conditions
        )

    @staticmethod
    def from_counts(counts: Dict[Union[str, int], int],
                    names: Union[str, List[str]]) -> 'DiscreteDistribution':
        """
        Return a new joint distribution from counts of random variable values.

        :param counts: Dictionary mapping values of random variables to the
                       number of observations.
        :param names: Name of each random variable.
        """
        sum_values = sum(counts.values())
        normalized = {k: v / sum_values for k, v in counts.items()}
        return DiscreteDistribution.from_dict(normalized, var_names=names)

    @staticmethod
    def from_observations(data: DataFrame) -> 'DiscreteDistribution':
        """
        Create a new discrete distribution based on the counts of items in the
        given data.

        :param data: DataFrame where each column represents a discrete random
                     variable, and each row represents an observation.
        """
        prob_data: Series = (
            data.groupby(list(data.columns)).size() / len(data)
        ).rename('p')
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
    def cond_var_names(self) -> List[str]:
        """
        Return a list of the names of all conditional variables.
        """
        return self._cond_var_names

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
        Return a name for the distribution based on the names of the joint
        variables and the names, conditional comparators and values of the given
        variables.
        """
        str_joints = ','.join(self._joints)
        strs_conditions = []
        for cond_var in self._cond_var_names:
            strs_conditions.append(cond_var)
        for cond_var, cond_val in self.given_conditions.items():
            strs_conditions.append(
                cond_name_and_symbol(cond_var, cond_val, self.var_names)
            )
        str_conditions = (
            '|' + ','.join(strs_conditions) if strs_conditions else ''
        )
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
            raise ValueError(
                'Must pass fewer than total number '
                'of joint variables as margins.'
            )
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
            raise ValueError(
                'All conditioning variables must be joint variables.'
            )
        if not len(cond_var_names) < len(self._joints):
            raise ValueError(
                'Cannot condition on all joint variables simultaneously.'
            )
        # calculate conditional table
        data = condition(self._data, *cond_var_names)
        from probability.discrete.conditional_table import ConditionalTable
        return ConditionalTable(
            data=data, cond_var_names=[cv for cv in cond_var_names]
        )

    def given(self, **given_conditions) -> 'DiscreteDistribution':
        """
        Condition on values of variables.

        :param given_conditions: Dict[{name}__{comparator}, value] for each
                                 conditioned variable.
        """
        # check input variables
        names_comps = set(given_conditions.keys())
        if not all([valid_name_comparator(name_comp, self._joints)
                    for name_comp in names_comps]):
            raise ValueError(
                'Given variables must be members of joint distribution.'
            )
        # calculate conditional distribution
        data = given(self._data, **given_conditions)
        cond_values = {**self._given_conditions, **given_conditions}
        return DiscreteDistribution(
            data=data,
            cond_var_names=self._cond_var_names,
            given_conditions=cond_values
        )

    def p(self, **joint_vals) -> float:
        """
        Return the value of the probability distribution at the values of the
        joint variables.

        :param joint_vals: Dict[name, value] for each joint variable to find
                           probability at.
        """
        # check input variables
        joint_arg_names = set([cond_name(joint_var, self._joints)
                               for joint_var in joint_vals.keys()])
        if not joint_arg_names == set(self._joints):
            raise ValueError(
                'Must specify a value (+condition) for each joint variable.'
            )
        # find probability of joint variable values
        return p(self._data, **joint_vals)

    # endregion

    # region operators

    @overload
    def __mul__(self, other: 'DiscreteDistribution') -> 'DiscreteDistribution':
        """
        Multiply the joint distribution e.g. `P(A)` by another
        DiscreteDistribution e.g. P(B) and return a new joint distribution
        i.e. `P(A,B)`. Assumes that P(A) and P(B) are independent such that
        P(A|B) = P(A)

        :param other: DiscreteDistribution to multiply by.
        """
        pass

    @overload
    def __mul__(self, other: 'ConditionalTable') -> 'DiscreteDistribution':
        """
        Multiply the joint distribution e.g. `P(B)` by a ConditionalTable
        conditioned over this distribution's joint variables e.g. `P(A|B)` and
        return a new joint distribution i.e. `P(A)` over the ConditionalTable's
        joint variables i.e. `A`.

        :param other: ConditionalTable with conditional variables matching this
                      distribution's joint variables.
        """
        pass

    def __mul__(self, other):

        from probability.discrete import ConditionalTable
        if isinstance(other, ConditionalTable):
            if self.given_conditions:
                raise ValueError('DiscreteDistribution cannot have '
                                 'any conditional variables.')
            if set(self.joints) != set(other.cond_vars):
                raise ValueError(
                    'Joint variables of DiscreteDistribution must be same as'
                    ' conditional variables of ConditionalTable.'
                )
            data = multiply(conditional=other.data, marginal=self.data)
            return DiscreteDistribution(data=data)
        elif isinstance(other, DiscreteDistribution):
            if (
                    self.given_conditions or other.given_conditions or
                    self.cond_var_names or other.cond_var_names
            ):
                raise ValueError('Neither DiscreteDistribution should have '
                                 'conditional variables or values.')
            if set(self.var_names).intersection(other.var_names):
                raise ValueError('DiscreteDistributions have shared variables.')
            keys = []
            values = []
            for ix_1, p_1 in self._data.iteritems():
                for ix_2, p_2 in other._data.iteritems():
                    if not isinstance(ix_1, tuple):
                        ix_1 = (ix_1,)
                    if not isinstance(ix_2, tuple):
                        ix_2 = (ix_2,)
                    keys.append((*ix_1, *ix_2))
                    values.append(p_1 * p_2)
            data = Series(
                data=values,
                index=MultiIndex.from_tuples(
                    tuples=keys,
                    names=self._data.index.names + other._data.index.names
                ))
            return DiscreteDistribution(data=data)

    def __rmul__(self, other: 'ConditionalTable') -> 'DiscreteDistribution':
        return self * other

    def __truediv__(self, other: 'DiscreteDistribution') -> 'ConditionalTable':

        if not set(other.joints).issubset(self.joints):
            raise ValueError(
                'All joint variables in denominator distribution must be '
                'present in numerator distribution to divide.'
            )
        if not len(other.joints) < len(self.joints):
            raise ValueError('Denominator distribution must contain fewer '
                             'joint variables than numerator distribution.')
        from probability.discrete.conditional_table import ConditionalTable
        return ConditionalTable(data=self._data / other._data,
                                cond_var_names=other.joints)

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
