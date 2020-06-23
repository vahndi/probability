from typing import List

from pandas import Series, DataFrame, MultiIndex
from pgmpy.factors.discrete import TabularCPD


class CPT(object):
    """
    Conditional Probability Table
    """
    def __init__(self, tabular_cpd: TabularCPD):

        self._tabular_cpd: TabularCPD = tabular_cpd

    @staticmethod
    def from_data_frame(data: DataFrame) -> 'CPT':
        """
        Create a new CPT from a DataFrame where:
            * index contains states of joint variables
            * columns contains states of conditional variables
            * index and column names are names of variables
            * values are probabilities of index states given column states

        :param data: DataFrame to get values, variables and states from.
        """
        variable = data.index.name
        variable_card = data.index.nunique()
        if isinstance(data.columns, MultiIndex):
            evidence = data.columns.names
            evidence_card = data.columns.levshape
        else:
            evidence = data.columns.name
            evidence_card = data.columns.nunique()
        state_names = {
            variable: data.index.unique().to_list()
        }
        for i_level, name in zip(
            range(data.columns.nlevels),
            data.columns.names
        ):
            state_names[name] = (
                data.columns.get_level_values(i_level).unique().to_list()
            )

        cpt = TabularCPD(
            variable=variable,
            variable_card=variable_card,
            values=data.values,
            evidence=evidence,
            evidence_card=evidence_card,
            state_names=state_names
        )

        return CPT(tabular_cpd=cpt)

    @property
    def tabular_cpd(self) -> TabularCPD:
        """
        Return the wrapped pgmpy TabularCPD.
        """
        return self._tabular_cpd

    @property
    def variable(self) -> str:
        """
        Return the name of the probability variable.
        """
        return self._tabular_cpd.variable

    def evidence(self) -> List[str]:
        """
        Return the name(s) of the conditioned variables.
        """
        return self._tabular_cpd.get_evidence()
