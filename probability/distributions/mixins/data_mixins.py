from collections import OrderedDict
from typing import Union, List, Optional, Tuple, TypeVar, Any

from numpy import log
from pandas import Series, DataFrame, concat, merge
from scipy.stats import entropy

from mpl_format.axes import AxesFormatter
from mpl_format.compound_types import Color, FontSize
from mpl_format.enums import FONT_SIZE
from mpl_format.utils.number_utils import format_as_percent, format_as_integer
from probability.custom_types.external_custom_types import FloatArray1d
from probability.distributions import Beta
from probability.distributions.conjugate.dirichlet_multinomial_conjugate import \
    DirichletMultinomialConjugate
from probability.distributions.continuous.beta_frame import BetaFrame
from probability.distributions.continuous.beta_series import BetaSeries

DDM = TypeVar('DDM', bound='DataDistributionMixin')
DCM = TypeVar('DCM', bound='DataCategoriesMixin')
DNM = TypeVar('DNM', bound='DataNumericMixin')


class DataDistributionMixin(object):

    _data: Series

    @property
    def data(self) -> Series:
        """
        Return the underlying data used to construct the Distribution.
        """
        return self._data

    @property
    def name(self):
        """
        Return the name of the Series of data.
        """
        return self._data.name

    def rename(self: DDM, name: str) -> DDM:
        """
        Rename the Series of data.
        """
        return type(self)(data=self._data.rename(name))

    def filter_to(self: DDM, other: 'DataDistributionMixin') -> DDM:
        """
        Filter the data to the common indices with the other distribution.
        """
        merged = merge(left=self.data, right=other.data,
                       left_index=True, right_index=True)
        data = merged.iloc[:, 0]
        return type(self)(data=data)

    def __len__(self):

        return len(self._data)


class DataDiscreteMixin(object):

    name: str
    _data: Series
    _categories: List[Any]

    def counts(self) -> Series:
        """
        Return a Series with the count of each category.
        """
        value_counts = self._data.value_counts().reindex(self._categories)
        return value_counts.fillna(0)

    def pmf(self) -> Series:
        """
        Return a Series with the probability of each category.
        """
        counts = self.counts()
        return counts / counts.sum()

    def plot_bars(
            self,
            color: Color = 'k',
            pct_font_size: int = FONT_SIZE.medium,
            axf: Optional[AxesFormatter] = None
    ) -> AxesFormatter:
        """
        Plot a bar plot of the counts of each category.

        :param color: Color of the bars.
        :param pct_font_size: Font size for percentage labels.
        :param axf: Optional AxesFormatter instance.
        """
        axf = axf or AxesFormatter()
        counts = self.counts()
        counts.plot.bar(ax=axf.axes, color=color)
        percents = 100 * self.pmf()
        axf.add_text(
            x=range(len(counts)), y=counts,
            text=percents.map(lambda p: f'{p: .1f}%'),
            h_align='center', v_align='bottom',
            font_size=pct_font_size
        )
        axf.y_axis.set_format_integer()
        return axf

    def plot_comparison_bars(
            self,
            other: 'DataDiscreteMixin',
            absolute: bool = False,
            color: Tuple[Color, Color] = ('C0', 'C1'),
            width: float = 0.5,
            label_pcts: bool = True,
            label_counts: bool = False,
            label_size: Optional[FontSize] = FONT_SIZE.medium,
            axf: Optional[AxesFormatter] = None
    ) -> AxesFormatter:
        """
        Plot a comparison of the 2 ordinals, with outlines around bars that
        are significantly higher or lower than others.

        :param other: Another Ordinal with the same categories.
        :param absolute: Whether to plot bar heights as absolute values or
                        percentages.
        :param color: Color for each set of bars.
        :param width: Total width of each pair of bars.
        :param label_pcts: Whether to add percentage labels.
        :param label_counts: Whether to add count labels.
        :param label_size: Font size for bar labels.
        :param axf: Optional AxesFormatter instance.
        """
        # validation
        if self.name == other.name:
            raise ValueError(
                'Distributions must have different names in order to compare.')
        self_cats, other_cats = set(self._categories), set(other._categories)
        if self_cats != other_cats:
            str_warning = f'WARNING: Distributions contain different categories'
            unique_1 = self_cats.difference(other_cats)
            unique_2 = other_cats.difference(self_cats)
            if unique_1:
                str_warning += f'\nOnly in {self.name}: {", ".join(unique_1)}'
            if unique_2:
                str_warning += f'\nOnly in {other.name}: {", ".join(unique_2)}'
            print(str_warning)
        # get data
        self_counts = self._data.value_counts().reindex(self._categories)
        other_counts = other._data.value_counts().reindex(self._categories)
        count_data = concat([self_counts, other_counts], axis=1)
        pct_data = count_data / count_data.sum()
        # plot bars
        axf = axf or AxesFormatter()
        if absolute:
            plot_data = count_data
        else:
            plot_data = pct_data
        plot_data.plot.bar(ax=axf.axes, color=color, width=width)
        # add labels
        for o, ordinal in enumerate(plot_data.columns):
            for i, ix in enumerate(plot_data[ordinal].index):
                if not label_pcts and not label_counts:
                    continue
                label_x = i - width / 4 + o * width / 2
                label_y = plot_data.loc[ix, ordinal]
                texts = []
                if label_pcts:
                    texts.append(format_as_percent(pct_data.loc[ix, ordinal]))
                if label_counts:
                    texts.append(format_as_integer(count_data.loc[ix, ordinal]))
                text = ' | '.join(texts)
                axf.add_text(label_x, label_y, text,
                             font_size=label_size,
                             h_align='center', v_align='bottom')
        # format y-axis
        if not absolute:
            axf.y_axis.set_format_percent()
        else:
            axf.y_axis.set_format_integer()

        return axf


class DataCategoriesMixin(object):

    _categories: List[Union[bool, str]]
    _data: Series
    _ordered: bool
    name: str

    @property
    def categories(self) -> list:
        """
        Return the names of the data categories.
        """
        return self._categories

    @property
    def num_categories(self) -> int:
        """
        Return the number of data categories.
        """
        return len(self._categories)

    def drop(self: DCM, categories: Union[str, List[str]]) -> DCM:
        """
        Drop one or more categories from the underlying data.

        :param categories: Categories to drop.
        """
        if isinstance(categories, str):
            categories = [categories]
        data = self._data.loc[~self._data.isin(categories)]
        new_cats = [cat for cat in self._categories
                    if cat not in categories]
        data = data.cat.set_categories(
            new_categories=new_cats,
            ordered=self._ordered
        )
        return type(self)(data=data)

    def keep(self: DCM, categories: Union[str, List[str]]) -> DCM:
        """
        Drop all the categories from the data not in the one(s) given.

        :param categories: Categories to keep.
        """
        if isinstance(categories, str):
            categories = [categories]
        data = self._data.loc[self._data.isin(categories)]
        new_cats = [cat for cat in self._categories if cat in categories]
        data = data.cat.set_categories(
            new_categories=new_cats,
            ordered=self._ordered
        )
        return type(self)(data=data)

    def rename_categories(
            self: DCM,
            new_categories: Union[list, dict]
    ) -> DCM:
        """
        Return a new Ordinal with its categories renamed.
        """
        if (
                isinstance(new_categories, list) or
                (isinstance(new_categories, dict) and
                 len(new_categories.values()) ==
                 len(set(new_categories.values())))
        ):
            # one to one
            return type(self)(
                data=self._data.cat.rename_categories(new_categories)
            )
        else:
            # many to one
            data = self._data.replace(new_categories).astype('category')
            categories = list(OrderedDict.fromkeys(new_categories.values()))
            data = data.cat.set_categories(
                new_categories=categories,
                ordered=self._ordered
            )
            return type(self)(data=data)

    def pmf_betas(
            self,
            alpha: Optional[Union[FloatArray1d, dict, float]] = None
    ) -> BetaSeries:
        """
        Return a BetaSeries with the probability of each category as a
        Beta distribution.

        :param alpha: Value(s) for the α hyper-parameter of the prior Dirichlet
                      distribution. Defaults to Uniform distribution,
        """
        dirichlet = DirichletMultinomialConjugate.infer_posterior(
            data=self._data, alpha=alpha
        )
        return BetaSeries(Series({
            name: dirichlet[name]
            for name in dirichlet.names
        }))


class DataNumericMixin(object):

    _data: Series

    def where_eq(self: DNM, value: Union[int, float]) -> DNM:

        return type(self)(data=self._data.loc[self._data == value])

    def where_ne(self: DNM, value: Union[int, float]) -> DNM:

        return type(self)(data=self._data.loc[self._data != value])

    def where_gt(self: DNM, value: Union[int, float]) -> DNM:

        return type(self)(data=self._data.loc[self._data > value])

    def where_lt(self: DNM, value: Union[int, float]) -> DNM:

        return type(self)(data=self._data.loc[self._data < value])

    def where_ge(self: DNM, value: Union[int, float]) -> DNM:

        return type(self)(data=self._data.loc[self._data >= value])

    def where_le(self: DNM, value: Union[int, float]) -> DNM:

        return type(self)(data=self._data.loc[self._data <= value])


class DataMinMixin(object):

    _data: Series

    def min(self) -> Union[float, int, str]:
        """
        Return the smallest value in the data.
        """
        return self._data.min()


class DataMaxMixin(object):

    _data: Series

    def max(self) -> Union[float, int, str]:
        """
        Return the largest value in the data.
        """
        return self._data.max()


class DataMeanMixin(object):

    _data: Series

    def mean(self) -> float:
        """
        Return the mean value of the data.
        """
        return self._data.mean()


class DataMedianMixin(object):

    _data: Series

    def median(self) -> Union[int, float]:
        """
        Return the median value of the data.
        """
        return self._data.median()


class DataStdMixin(object):

    _data: Series

    def std(self) -> float:
        """
        Return the median value of the data.
        """
        return self._data.std()


class DataModeMixin(object):

    _data: Series

    def mode(self) -> Union[int, float, str,
                            List[int], List[float], List[str]]:
        """
        Return the most frequently occurring value(s) in the data.
        """
        mode = self._data.mode()
        if len(mode) > 1:
            return mode.to_list()
        else:
            return mode[0]


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
            condition: 'DataProbabilityTableMixin'
    ) -> DataFrame:
        """
        Return the conditional probability of each category given different
        values of condition.
        """
        self_name = self._data.name
        other_name = condition._data.name
        data = concat([condition._data, self._data], axis=1)
        joint = data[[other_name, self_name]].value_counts()
        marginal = data[other_name].value_counts()
        marginal.index.name = other_name
        return (joint / marginal).unstack()

    def cpt(
            self,
            condition: 'DataProbabilityTableMixin'
    ) -> DataFrame:
        """
        Return the conditional probability of each category given different
        values of condition.

        Alias for conditional_probability_table.
        """
        return self.conditional_probability_table(condition=condition)

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
