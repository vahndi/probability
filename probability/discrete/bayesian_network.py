from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.d_separation import is_d_separator
import pandas as pd
from pandas import DataFrame, MultiIndex, Series

from probability.discrete.conditional import Conditional
from probability.discrete.discrete import Discrete
from probability.plotting import new_subplots


class BayesianNetwork(object):
    """
    A Discrete Bayesian Network (Belief Network).

    Represents a Directed Acyclic Graph (DAG) whose nodes represent discrete
    random variables and edges represent direct probabilistic dependencies.
    Each node has an associated Conditional Probability Table (CPT):
      - Nodes without parents have a marginal `Discrete` prior distribution P(X).
      - Nodes with parents have a `Conditional` distribution P(X | Parents(X)).

    Supports:
      - Variable Elimination for exact marginal and conditional inference P(T | E=e).
      - D-separation and conditional independence queries (net.is_independent).
      - Network visualization (.draw).
      - Full joint distribution (.joint).
    """

    def __init__(
            self,
            nodes: Optional[Iterable[Union[Discrete, Conditional]]] = None
    ):
        """
        Create a new Bayesian Network.

        :param nodes: Optional collection of Discrete (root) and Conditional (child) CPTs.
        """
        self._graph: nx.DiGraph = nx.DiGraph()
        self._cpts: Dict[str, Union[Discrete, Conditional]] = {}
        self._states: Dict[str, list] = {}

        if nodes is not None:
            self.add_nodes(*nodes)

    @property
    def graph(self) -> nx.DiGraph:
        """The underlying NetworkX directed graph."""
        return self._graph

    @property
    def nodes(self) -> List[str]:
        """List of variable names in the network."""
        return list(self._graph.nodes)

    @property
    def edges(self) -> List[Tuple[str, str]]:
        """List of directed edges (parent, child)."""
        return list(self._graph.edges)

    def parents(self, var: str) -> List[str]:
        """Return parent variable names for `var`."""
        if var not in self._graph:
            raise KeyError(f"Variable '{var}' not in network.")
        return list(self._graph.predecessors(var))

    def children(self, var: str) -> List[str]:
        """Return child variable names for `var`."""
        if var not in self._graph:
            raise KeyError(f"Variable '{var}' not in network.")
        return list(self._graph.successors(var))

    def cpt(self, var: str) -> Union[Discrete, Conditional]:
        """Return the CPT (Discrete or Conditional) for `var`."""
        if var not in self._cpts:
            raise KeyError(f"CPT for variable '{var}' not found.")
        return self._cpts[var]

    def states(self, var: str) -> list:
        """Return list of valid state values for `var`."""
        if var not in self._states:
            raise KeyError(f"Variable '{var}' not in network.")
        return self._states[var]

    def add_node(self, node: Union[Discrete, Conditional]) -> 'BayesianNetwork':
        """
        Add a node with its CPT to the network.

        :param node: Discrete (if no parents) or Conditional (if has parents).
        """
        if isinstance(node, Discrete):
            if len(node.variables) != 1:
                raise ValueError("Discrete node in BayesianNetwork must have exactly 1 variable.")
            var_name = node.variables[0]
            self._graph.add_node(var_name)
            self._cpts[var_name] = node
            self._states[var_name] = list(node.states[var_name])

        elif isinstance(node, Conditional):
            if len(node.joint_variables) != 1:
                raise ValueError("Conditional node in BayesianNetwork must have exactly 1 joint variable.")
            var_name = node.joint_variables[0]
            self._graph.add_node(var_name)
            for parent in node.conditional_variables:
                self._graph.add_edge(parent, var_name)
            self._cpts[var_name] = node
            self._states[var_name] = list(node.states[var_name])
            for parent in node.conditional_variables:
                if parent in node.states and parent not in self._states:
                    self._states[parent] = list(node.states[parent])

        else:
            raise TypeError("Node must be an instance of Discrete or Conditional.")

        if not nx.is_directed_acyclic_graph(self._graph):
            raise ValueError(f"Adding node '{var_name}' created a cycle in the Bayesian Network.")

        return self

    def add_nodes(self, *nodes: Union[Discrete, Conditional]) -> 'BayesianNetwork':
        """Add multiple CPT nodes to the network."""
        for node in nodes:
            self.add_node(node)
        return self

    def is_independent(
            self,
            x: Union[str, Iterable[str]],
            y: Union[str, Iterable[str]],
            given: Optional[Union[str, Iterable[str]]] = None
    ) -> bool:
        """
        Test whether variables X and Y are conditionally independent given Z (d-separation).

        :param x: Variable name or collection of names X.
        :param y: Variable name or collection of names Y.
        :param given: Optional variable name or collection of conditioning names Z.
        :return: True if X is conditionally independent of Y given Z, else False.
        """
        x_set = {x} if isinstance(x, str) else set(x)
        y_set = {y} if isinstance(y, str) else set(y)
        z_set: Set[str] = set()
        if given is not None:
            z_set = {given} if isinstance(given, str) else set(given)

        for v in x_set | y_set | z_set:
            if v not in self._graph:
                raise KeyError(f"Variable '{v}' not found in network.")

        return is_d_separator(self._graph, x_set, y_set, z_set)

    def d_separated(
            self,
            x: Union[str, Iterable[str]],
            y: Union[str, Iterable[str]],
            given: Optional[Union[str, Iterable[str]]] = None
    ) -> bool:
        """Alias for `is_independent`."""
        return self.is_independent(x, y, given=given)

    @staticmethod
    def _to_factor(cpt: Union[Discrete, Conditional]) -> Series:
        """Convert a Discrete or Conditional CPT to a factor Series."""
        if isinstance(cpt, Discrete):
            s = cpt.data.copy()
            if not isinstance(s.index, MultiIndex):
                s.index = MultiIndex.from_arrays([s.index], names=cpt.variables)
            return s
        else:
            return cpt.data.stack(cpt.conditional_variables)

    @staticmethod
    def _factor_product(f1: Union[Series, float], f2: Union[Series, float]) -> Union[Series, float]:
        """Multiply two factors."""
        if isinstance(f1, (int, float)) and isinstance(f2, (int, float)):
            return float(f1) * float(f2)
        if isinstance(f1, (int, float)):
            return f2 * float(f1)
        if isinstance(f2, (int, float)):
            return f1 * float(f2)

        v1 = list(f1.index.names)
        v2 = list(f2.index.names)
        common = [v for v in v1 if v in v2]

        df1 = f1.reset_index(name='_val1')
        df2 = f2.reset_index(name='_val2')
        if common:
            merged = pd.merge(df1, df2, on=common)
        else:
            merged = pd.merge(df1, df2, how='cross')

        merged['_p'] = merged['_val1'] * merged['_val2']
        all_vars = v1 + [v for v in v2 if v not in v1]
        return merged.set_index(all_vars)['_p']

    @staticmethod
    def _factor_marginalize(factor: Union[Series, float], var: str) -> Union[Series, float]:
        """Sum out variable `var` from factor."""
        if isinstance(factor, (int, float)):
            return factor
        rem_vars = [v for v in factor.index.names if v != var]
        if rem_vars:
            return factor.groupby(level=rem_vars).sum()
        else:
            return float(factor.sum())

    @staticmethod
    def _factor_slice(factor: Union[Series, float], var: str, val: Any) -> Union[Series, float]:
        """Instantiate evidence var = val into factor."""
        if isinstance(factor, (int, float)):
            return factor
        if var in factor.index.names:
            rem_names = [v for v in factor.index.names if v != var]
            res = factor.xs(val, level=var)
            if not rem_names:
                if isinstance(res, Series):
                    return float(res.iloc[0])
                return float(res)
            if len(rem_names) == 1 and not isinstance(res.index, MultiIndex):
                res.index = MultiIndex.from_arrays([res.index], names=rem_names)
            return res
        return factor

    def joint(self) -> Discrete:
        """
        Compute the full joint distribution P(X_1, ..., X_N) as a Discrete distribution.
        """
        order = list(nx.topological_sort(self._graph))
        prod: Optional[Series] = None
        for var in order:
            f = self._to_factor(self._cpts[var])
            prod = f if prod is None else self._factor_product(prod, f)

        prod = prod / prod.sum()
        return Discrete(
            data=prod,
            variables=list(prod.index.names),
            states=self._states
        )

    def query(
            self,
            target: Union[str, Iterable[str]],
            evidence: Optional[Dict[str, Any]] = None,
            elimination_order: Optional[List[str]] = None
    ) -> Discrete:
        """
        Exact inference via Variable Elimination: compute P(Target | Evidence).

        :param target: Target variable name, or collection of target variables.
        :param evidence: Optional dict of observed evidence {var: value}.
        :param elimination_order: Optional explicit ordering of variables to eliminate.
        :return: Discrete posterior distribution over target variable(s).
        """
        target_set = {target} if isinstance(target, str) else set(target)
        evidence = evidence or {}
        evidence_set = set(evidence.keys())

        # Validate variables
        for v in target_set | evidence_set:
            if v not in self._graph:
                raise KeyError(f"Variable '{v}' not found in network.")
        if target_set & evidence_set:
            overlap = target_set & evidence_set
            raise ValueError(f"Variables cannot be both target and evidence: {overlap}")

        # 1. Collect all initial factors
        factors: List[Union[Series, float]] = [
            self._to_factor(self._cpts[node])
            for node in self._graph.nodes
        ]

        # 2. Instantiate evidence
        for ev_var, ev_val in evidence.items():
            factors = [self._factor_slice(f, ev_var, ev_val) for f in factors]

        # 3. Determine elimination order for hidden variables
        hidden_vars = [
            node for node in nx.topological_sort(self._graph)
            if node not in target_set and node not in evidence_set
        ]

        if elimination_order is not None:
            order = [v for v in elimination_order if v in hidden_vars]
        else:
            # Min-neighbors heuristic
            order = []
            remaining_hidden = set(hidden_vars)
            while remaining_hidden:
                # Count current factor scope size for each candidate
                scores = {}
                for v in remaining_hidden:
                    scope = set()
                    for f in factors:
                        if isinstance(f, Series) and v in f.index.names:
                            scope.update(f.index.names)
                    scores[v] = len(scope)
                best_var = min(scores, key=scores.get)
                order.append(best_var)
                remaining_hidden.remove(best_var)

        # 4. Eliminate hidden variables
        for var in order:
            relevant = [f for f in factors if isinstance(f, Series) and var in f.index.names]
            irrelevant = [f for f in factors if not (isinstance(f, Series) and var in f.index.names)]

            if not relevant:
                continue

            # Multiply relevant factors
            product_factor = relevant[0]
            for f in relevant[1:]:
                product_factor = self._factor_product(product_factor, f)

            # Marginalize out var
            summed_factor = self._factor_marginalize(product_factor, var)
            factors = irrelevant + [summed_factor]

        # 5. Multiply remaining factors containing target
        final_factor: Union[Series, float] = factors[0]
        for f in factors[1:]:
            final_factor = self._factor_product(final_factor, f)

        if isinstance(final_factor, (int, float)):
            # All variables were evidence, probability of evidence
            return final_factor

        # Normalize over target variables
        total = final_factor.sum()
        if total == 0:
            raise ValueError("Zero probability evidence observed.")
        final_factor = final_factor / total

        return Discrete(
            data=final_factor,
            variables=list(final_factor.index.names),
            states={v: self._states[v] for v in final_factor.index.names if v in self._states}
        )

    def p(self, *args, given: Optional[Dict[str, Any]] = None, **kwargs) -> float:
        """
        Calculate scalar conditional or marginal probability.

        Flexible syntax:
          - net.p(burglary='yes', given={'alarm': 'yes'})
          - net.p('burglary', 'yes', given={'alarm': 'yes'})
          - net.p(burglary='yes', alarm='yes')
        """
        evidence = dict(given or {})
        target_var: Optional[str] = None
        target_val: Any = None

        if len(args) == 2 and isinstance(args[0], str):
            target_var, target_val = args[0], args[1]
        elif len(args) == 1 and isinstance(args[0], str):
            target_var = args[0]
        elif kwargs:
            # If target specified as kwarg
            kw_items = list(kwargs.items())
            if len(kw_items) == 1 and given is not None:
                target_var, target_val = kw_items[0]
            elif len(kw_items) >= 1:
                # First is target, rest are evidence if given was not passed
                target_var, target_val = kw_items[0]
                for k, v in kw_items[1:]:
                    evidence[k] = v

        if target_var is None:
            raise ValueError("Must specify target variable and value.")

        dist = self.query(target=target_var, evidence=evidence)
        if target_val is not None:
            return float(dist.p(**{target_var: target_val}))
        else:
            return dist

    def draw(
            self,
            ax: Optional[plt.Axes] = None,
            layout: str = 'spring',
            node_color: str = '#A0CBE2',
            node_size: int = 2400,
            font_size: int = 11,
            font_weight: str = 'bold',
            edge_color: str = '#4A4A4A',
            arrowsize: int = 20,
            **kwargs
    ) -> plt.Axes:
        """
        Draw the directed acyclic graph structure of the Bayesian Network.

        :param ax: Matplotlib axes to plot onto (creates new figure if None).
        :param layout: Graph layout algorithm ('spring', 'shell', 'spectral', 'kamada_kawai').
        :param node_color: Node background color.
        :param node_size: Size of nodes.
        :param font_size: Font size for variable labels.
        :param font_weight: Font weight for variable labels.
        :param edge_color: Arrow color.
        :param arrowsize: Size of arrowheads.
        """
        if ax is None:
            fig, ax = new_subplots(1, 1)

        if layout == 'spring':
            pos = nx.spring_layout(self._graph, seed=42)
        elif layout == 'shell':
            pos = nx.shell_layout(self._graph)
        elif layout == 'spectral':
            pos = nx.spectral_layout(self._graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self._graph)
        else:
            pos = nx.spring_layout(self._graph, seed=42)

        nx.draw_networkx_nodes(
            self._graph, pos, ax=ax,
            node_color=node_color, node_size=node_size, **kwargs
        )
        nx.draw_networkx_edges(
            self._graph, pos, ax=ax,
            edge_color=edge_color, arrowsize=arrowsize,
            arrowstyle='-|>', node_size=node_size, **kwargs
        )
        nx.draw_networkx_labels(
            self._graph, pos, ax=ax,
            font_size=font_size, font_weight=font_weight
        )

        ax.set_title("Bayesian Network (DAG)")
        ax.axis('off')
        return ax

    def __repr__(self) -> str:
        nodes_str = ', '.join(self.nodes)
        return f"BayesianNetwork(nodes=[{nodes_str}], edges={len(self.edges)})"
