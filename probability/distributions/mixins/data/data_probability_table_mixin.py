from typing import Optional

from pandas import Series, concat, DataFrame

from probability.distributions import Beta
from probability.distributions.continuous.beta_frame import BetaFrame


class DataProbabilityTableMixin(object):

    _data: Series

    def joint_count_table(
            self,
            other: 'DataProbabilityTableMixin'
    ):
        """
        Return the joint count of each category with different values of other.
        """
        self_name = self._data.name
        other_name = other._data.name
        data = concat([other._data, self._data], axis=1)
        joint = data[[other_name, self_name]].value_counts()
        return joint.unstack()

    def jct(
            self,
            other: 'DataProbabilityTableMixin'
    ):
        """
        Return the joint count of each category with different values of other.

        Alias for joint_count_table.
        """
        return self.joint_count_table(other=other)

    def joint_probability_table(
            self,
            other: 'DataProbabilityTableMixin'
    ):
        """
        Return the joint probability of each category with different values of
        other.
        """
        jct = self.jct(other)
        return jct / jct.sum().sum()

    def jpt(
            self,
            other: 'DataProbabilityTableMixin'
    ):
        """
        Return the joint probability of each category with different values of
        other.

        Alias for joint_probability_table.
        """
        return self.joint_probability_table(other)

    def conditional_probability_table(
            self,
            condition: 'DataProbabilityTableMixin',
            weights=None
    ) -> DataFrame:
        """
        Return the conditional probability of each category given different
        values of condition.

        :param condition: Other categorical to condition on.
        :param weights: Optional distribution of weights.
        """
        self_name = self._data.name
        other_name = condition._data.name
        if weights is None:
            data = concat([condition._data, self._data], axis=1)
            joint = data[[other_name, self_name]].value_counts()
            marginal = data[other_name].value_counts()
        else:
            weights_name = 'weights'
            while weights_name in [self_name, other_name]:
                weights_name += '_'
            data = concat([
                condition._data,
                self._data,
                weights._counts.rename(weights_name)
            ], axis=1)
            joint = data.groupby([other_name, self_name])[weights_name].sum()
            marginal = data.groupby(other_name)[weights_name].sum()
        marginal.index.name = other_name
        return (joint / marginal).unstack()

    def cpt(
            self,
            condition: 'DataProbabilityTableMixin',
            weights=None
    ) -> DataFrame:
        """
        Return the conditional probability of each category given different
        values of condition.

        Alias for conditional_probability_table.
        """
        return self.conditional_probability_table(
            condition=condition,
            weights=weights
        )

    def conditional_beta_table(
            self,
            condition: 'DataProbabilityTableMixin'
    ) -> BetaFrame:
        """
        Return the conditional probability of each category given different
        values of condition.
        """
        counts = self.joint_count_table(condition)
        row_sums = counts.sum(axis=1).to_list()
        beta_dicts = []
        for r in range(len(row_sums)):
            beta_dicts.append(
                counts.iloc[r].map(
                    lambda count: Beta(count, row_sums[r] - count)
                )
            )
        return BetaFrame(DataFrame(beta_dicts))