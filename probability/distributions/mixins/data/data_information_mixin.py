from typing import Tuple, Optional

from numpy import log
from pandas import Series, merge, DataFrame, concat
from scipy.stats import entropy


class DataInformationMixin(object):
    """
    References
    ----------
    [1] Evaluating accuracy of community detection using the relative normalized
    mutual information, Pan Zhang https://arxiv.org/pdf/1501.03844.pdf
    [2] Probabilistic Machine Learning: An Introduction, K. Murphy
    """
    _data: Series

    def _get_shared_data(
            self,
            other: 'DataInformationMixin'
    ) -> Tuple[Series, Series]:

        merged = merge(left=self._data, right=other._data,
                       left_index=True, right_index=True)
        self_data = merged.iloc[:, 0]
        other_data = merged.iloc[:, 1]
        if len(self_data) < len(self._data):
            print(
                f'WARNING: DataInformationMixin operating on subset of size '
                f'{len(self_data)} of {len(self._data)} '
                f'({100 * len(self_data) / len(self._data): 0.1f}%)'
            )
        return self_data, other_data

    @staticmethod
    def _make_calc_frame(self_data: Series, other_data: Series) -> DataFrame:

        x_counts = self_data.value_counts().rename_axis('x')
        y_counts = other_data.value_counts().rename_axis('y')
        xy_counts = concat([
            self_data.rename('x'),
            other_data.rename('y')
        ], axis=1).value_counts()
        p_x = x_counts / x_counts.sum()
        p_y = y_counts / y_counts.sum()
        p_xy = xy_counts / xy_counts.sum()
        calc = p_xy.rename('p(x,y)').to_frame()
        calc['p(x)'] = p_x.reindex(
            calc.index.get_level_values('x')).to_list()
        calc['p(y)'] = p_y.reindex(
            calc.index.get_level_values('y')).to_list()
        calc['I(x,y)'] = calc['p(x,y)'] * (
                calc['p(x,y)'] / (calc['p(x)'] * calc['p(y)'])
        ).map(log)
        calc['H(x,y)'] = calc['p(x,y)'] * calc['p(x,y)'].map(log)
        calc['H(x|y)'] = calc['p(x,y)'] * (
                calc['p(x,y)'] / calc['p(x)']
        ).map(log)

        return calc

    def _calc_frame(self, other: 'DataInformationMixin') -> DataFrame:
        """
        Calculate various information measures of self and
        other and return as a DataFrame.
        """
        self_data, other_data = self._get_shared_data(other)
        return DataInformationMixin._make_calc_frame(self_data, other_data)

    def entropy(self) -> float:
        """
        Return the entropy of the distribution (self-information).
        """
        return entropy(self._data.value_counts())

    def mutual_information(self, other: 'DataInformationMixin') -> float:
        """
        In probability theory and information theory, the mutual information
        (MI) of two random variables is a measure of the mutual dependence
        between the two variables. More specifically, it quantifies the
        "amount of information" (in units such as shannons (bits), nats or
        hartleys) obtained about one random variable by observing the other
        random variable. The concept of mutual information is intimately linked
        to that of entropy of a random variable, a fundamental notion in
        information theory that quantifies the expected "amount of information"
        held in a random variable.

        https://en.wikipedia.org/wiki/Mutual_information
        """
        calc = self._calc_frame(other)
        return calc['I(x,y)'].sum()

    def mi(self, other: 'DataInformationMixin') -> float:
        """
        In probability theory and information theory, the mutual information
        (MI) of two random variables is a measure of the mutual dependence
        between the two variables. More specifically, it quantifies the
        "amount of information" (in units such as shannons (bits), nats or
        hartleys) obtained about one random variable by observing the other
        random variable. The concept of mutual information is intimately linked
        to that of entropy of a random variable, a fundamental notion in
        information theory that quantifies the expected "amount of information"
        held in a random variable.

        Alias for mutual_information.

        https://en.wikipedia.org/wiki/Mutual_information
        """
        return self.mutual_information(other=other)

    @staticmethod
    def _mutual_information(self_data: Series, other_data: Series):
        """
        Same as mutual_information but assumes data has already been filtered to
        a shared index.
        """
        calc = DataInformationMixin._make_calc_frame(self_data, other_data)
        return calc['I(x,y)'].sum()

    def conditional_entropy(self, other: 'DataInformationMixin') -> float:
        """
        In information theory, the conditional entropy quantifies the amount of
        information needed to describe the outcome of a random variable Y given
        that the value of another random variable X is known. Here, information
        is measured in shannons, nats, or hartleys. The entropy of Y conditioned
        on X is written as H(Y|X)}.

        https://en.wikipedia.org/wiki/Conditional_entropy
        """
        calc = self._calc_frame(other)
        return -calc['H(x|y)'].sum()

    def joint_entropy(self, other: 'DataInformationMixin') -> float:
        """
        In information theory, joint entropy is a measure of the uncertainty
        associated with a set of variables.

        https://en.wikipedia.org/wiki/Joint_entropy
        """
        calc = self._calc_frame(other)
        return -calc['H(x,y)'].sum()

    def relative_mutual_information(self, other: 'DataInformationMixin'):
        """
        Return the proportion of entropy in self explained by observing
        other. Relative here refers to the other distribution.
        """
        return self.mutual_information(other) / self.entropy()

    def rmi(self, other: 'DataInformationMixin'):
        """
        Return the proportion of entropy in self explained by observing
        other. Relative here refers to the other distribution.

        Alias for relative_mutual_information.
        """
        return self.relative_mutual_information(other)

    @staticmethod
    def _normalized_mutual_information(
            self_data: Series,
            other_data: Series,
            method: str = 'avg'
    ) -> float:
        """
        Same as normalized_mutual_information but assumes data has already been
        filtered to a shared index.
        """
        h_self = entropy(self_data.value_counts())
        h_other = entropy(other_data.value_counts())
        mi = DataInformationMixin._mutual_information(self_data, other_data)
        if method == 'max':
            return mi / min(h_self, h_other)
        elif method == 'avg':
            return 2 * mi / (h_self + h_other)
        else:
            raise ValueError("method must be one of {'max', 'avg'}")

    def normalized_mutual_information(
            self,
            other: 'DataInformationMixin',
            method: str = 'avg'
    ):
        """
        Return the maximum or average of the proportion of entropy explained
        by self by other and other by self.

        The 'avg' method is referenced in [1]
        The 'max' method is referenced in [2]
        """
        if method == 'max':
            return self.mutual_information(other) / min(
                self.entropy(), other.entropy()
            )
        elif method == 'avg':
            return 2 * self.mutual_information(other) / (
                self.entropy() + other.entropy()
            )
        else:
            raise ValueError("method must be one of {'max', 'avg'}")

    def nmi(
            self,
            other: 'DataInformationMixin',
            method: str = 'avg'
    ):
        """
        Return the maximum or average of the proportion of entropy explained
        by self by other and other by self.

        Alias for normalized_mutual_information.

        The 'avg' method is referenced in [1]
        The 'max' method is referenced in [2]
        """
        return self.normalized_mutual_information(
            other=other,
            method=method
        )

    def relative_normalized_mutual_information(
            self,
            other: 'DataInformationMixin',
            method: str = 'approx',
            samples: Optional[int] = None
    ):
        """
        Return the relative normalized mutual information (rNMI) as described in
        [1]. Relative here refers to a random distribution.

        :param other: Other distribution to compute rNMI with.
        :param method: One of {'approx', 'samples'}. 'approx' computes
                       NMI_random as described in [1] equation (9). 'samples'
                       computes  NMI_random as described in [1] equation (10).
                       Over estimation ε of rNMI using 'approx' for smaller
                       datasets in [1] depending on sample size n is around
                       n = 1,000 => ε = +24%,
                       n = 2,000 => ε = +12%,
                       n = 4,000 => ε = +6%,
                       n = 8,000 => ε = +3%
        :param samples: Number of random partitions to compute the expectation
                        of NMI(A, C) where method='samples'. Defaults to 10 as
                        described in [1].
        """
        h_a = self.entropy()
        h_b = other.entropy()
        self_data, other_data = self._get_shared_data(other)
        if method == 'approx':
            n = len(self_data)
            q_a = self_data.nunique()
            q_b = other_data.nunique()
            nmi_ac = (
                (q_a * q_b - q_a - q_b + 1) /
                ((h_a + h_b) * 2 * n)
            )
        elif method == 'samples':
            if samples is None:
                samples = 10
            nmi_ac_total = 0.0
            for s in range(samples):
                c_s = other_data.sample(frac=1).set_axis(other_data.index)
                calc = self._make_calc_frame(self_data, c_s)
                mi__a__c_s = calc['I(x,y)'].sum()
                nmi__a__c_s = 2 * mi__a__c_s / (h_a + h_b)
                nmi_ac_total += nmi__a__c_s
            nmi_ac = nmi_ac_total / samples
        else:
            raise ValueError("method must be one of {'approx', 'samples'}")

        nmi_ab = self._normalized_mutual_information(self_data, other_data)
        return nmi_ab - nmi_ac

    def rnmi(
            self,
            other: 'DataInformationMixin',
            method: str = 'approx',
            samples: Optional[int] = None
    ):
        """
        Return the relative normalized mutual information (rNMI) as described in
        [1].

        Alias for relative_normalized_mutual_information.

        :param other: Other distribution to compute rNMI with.
        :param method: One of {'approx', 'samples'}. 'approx' computes
                       NMI_random as described in [1] equation (9). 'samples'
                       computes  NMI_random as described in [1] equation (10)
        :param samples: Number of random partitions to compute the expectation
                        of NMI(A, C) where method='samples'. Defaults to 10 as
                        described in [1].
        """
        return self.relative_normalized_mutual_information(
            other=other,
            method=method,
            samples=samples
        )