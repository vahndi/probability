from typing import Optional, List, overload, Union

from matplotlib.axes import Axes
from mpl_format.axes.axis_utils import new_axes
from networkx import DiGraph, draw_networkx_nodes, \
    draw_networkx_edges, all_simple_paths, \
    descendants_at_distance, draw_networkx_labels
from pandas import DataFrame

from probability.models.decision_tree.nodes import DecisionNode, ChanceNode, \
    AmountNode


class DecisionTree(object):
    """
    A Probabilistic Decision Tree.
    """
    def __init__(self, max_depth: Optional[int] = None):
        """
        Create a new Probabilistic Decision Tree.
        """
        self._graph = DiGraph()
        self._num_decision_nodes = 0
        self._num_chance_nodes = 0
        self._num_amount_nodes = 0
        self._root_node: Optional[DecisionNode] = None
        self._max_depth: Optional[int] = max_depth

    @property
    def graph(self) -> DiGraph:
        """
        Return the wrapped networkx DiGraph object.
        """
        return self._graph

    def next_decision_number(self) -> int:
        """
        Return the number for the next DecisionNode and increase counter by 1.
        """
        self._num_decision_nodes += 1
        return self._num_decision_nodes

    def next_chance_number(self) -> int:
        """
        Return the number for the next ChanceNode and increase counter by 1.
        """
        self._num_chance_nodes += 1
        return self._num_chance_nodes

    def next_amount_number(self) -> int:
        """
        Return the number for the next AmountNode and increase counter by 1.
        """
        self._num_amount_nodes += 1
        return self._num_amount_nodes

    def _get_layout(self) -> dict:
        """
        Generate a layout for the Tree.
        """
        x_add = {
            DecisionNode: 0,
            ChanceNode: 1 / 3,
            AmountNode: 2 / 3
        }
        y_add = {
            DecisionNode: 0,
            ChanceNode: 1 / 3,
            AmountNode: 2 / 3
        }

        nodes = {}
        for depth in range(1, self._max_depth + 1):
            nodes[(DecisionNode, depth)] = self.decision_nodes(depth)
            nodes[(ChanceNode, depth)] = self.chance_nodes(depth)
            nodes[(AmountNode, depth)] = self.amount_nodes(depth)

        layout = {}
        for node in self._graph.nodes():
            node_type = type(node)
            node_list = nodes[(node_type, node.depth)]
            x = (node.depth + x_add[node_type]) / (self._max_depth * 3)
            y = (node_list.index(node) + y_add[node_type]) / len(node_list)
            layout[node] = [x, y]
        return layout

    def decision_nodes(self, depth: Optional[int] = None) -> List[DecisionNode]:
        """
        Return a list of all DecisionNodes in the DecisionTree.
        """
        nodes = [node for node in self._graph.nodes()
                 if isinstance(node, DecisionNode)]
        if depth is not None:
            nodes = [node for node in nodes if node.depth == depth]
        return nodes

    def chance_nodes(self, depth: Optional[int] = None) -> List[ChanceNode]:
        """
        Return a list of all ChanceNodes in the DecisionTree.
        """
        nodes = [node for node in self._graph.nodes()
                 if isinstance(node, ChanceNode)]
        if depth is not None:
            nodes = [node for node in nodes if node.depth == depth]
        return nodes

    def amount_nodes(self, depth: Optional[int] = None) -> List[AmountNode]:
        """
        Return a list of all AmountNodes in the DecisionTree.
        """
        nodes = [node for node in self._graph.nodes()
                 if isinstance(node, AmountNode)]
        if depth is not None:
            nodes = [node for node in nodes if node.depth == depth]
        return nodes

    def node(self, name: str):
        """
        Return the node with the given name.
        :param name: The name of the node.
        """
        return [node for node in self._graph.nodes()
                if node.name == name][0]

    def node_amounts(self) -> dict:
        """
        Return a dict mapping Nodes to their amounts.
        """
        return {
            node: node.str_amount
            for node in self._graph.nodes()
        }

    def node_names(self) -> dict:
        """
        Return a dict mapping Nodes to their names.
        """
        return {
            node: node.name
            for node in self._graph.nodes()
        }

    @overload
    def parent(self, node: AmountNode) -> ChanceNode:
        pass

    @overload
    def parent(self, node: ChanceNode) -> DecisionNode:
        pass

    @overload
    def parent(self, node: DecisionNode) -> Optional[ChanceNode]:
        pass

    def parent(self, node):
        return list(self._graph.predecessors(node))[0]

    @overload
    def children(self, node: DecisionNode) -> List[ChanceNode]:
        pass

    @overload
    def children(
            self, node: ChanceNode
    ) -> List[Union[DecisionNode, AmountNode]]:
        pass

    def children(self, node):
        return list(descendants_at_distance(self._graph, node, 1))

    def draw(
            self,
            node_labels: Optional[str] = None,
            ax: Optional[Axes] = None
    ) -> Axes:
        """
        Draw the DecisionTree.

        :param node_labels: One of {'name', 'amount', None}
        :param ax: Optional matplotlib axes.
        """
        ax = ax or new_axes()
        pos = self._get_layout()
        for nodes, node_shape, node_color in zip(
                (self.decision_nodes(),
                 self.chance_nodes(),
                 self.amount_nodes()),
                ('s', 'o', 'H'),
                ('r', 'b', 'g')
        ):
            draw_networkx_nodes(
                G=self._graph, pos=pos, ax=ax,
                nodelist=nodes,
                node_shape=node_shape, node_color=node_color, alpha=0.5,
            )
            if node_labels is not None:
                if node_labels == 'name':
                    labels = self.node_names()
                elif node_labels == 'amount':
                    labels = self.node_amounts()
                else:
                    raise ValueError()
                draw_networkx_labels(
                    G=self._graph, pos=pos, ax=ax,
                    labels=labels, font_size=10,
                )
        draw_networkx_edges(G=self._graph, pos=pos,
                            ax=ax, edge_color='gray')
        return ax

    def add_decision_node(self, decision_node: DecisionNode,
                          parent: Optional[ChanceNode] = None):

        if parent is None and self._root_node is not None:
            raise ValueError('Must give parent if tree already has a root node')
        self._graph.add_node(decision_node)
        if parent is not None:
            self._graph.add_edge(parent, decision_node)
        else:
            self._root_node = decision_node

    def add_chance_node(self, chance_node: ChanceNode,
                        parent: DecisionNode):

        self._graph.add_node(chance_node)
        self._graph.add_edge(parent, chance_node)

    def add_amount_node(self, amount_node: AmountNode,
                        parent: ChanceNode):

        self._graph.add_node(amount_node)
        self._graph.add_edge(parent, amount_node)

    def solve(self, minimize: bool = True):
        """
        Solve the Decision Tree.

        :param minimize: True to minimize amounts, i.e. amounts are costs, or
                         False to maximize amounts, i.e. amounts are rewards.
        """
        if minimize:
            opt_func = min
        else:
            opt_func = max

        # 1) at each end point of the tree write down the net total cost
        #    incurred if that end point is reached
        for amount_node in self.amount_nodes():
            path_to_node = list(all_simple_paths(
                self._graph, self._root_node, amount_node
            ))[0]
            total_amount = 0
            for node in path_to_node:
                if isinstance(node, ChanceNode):
                    total_amount += node.amount
            amount_node: AmountNode = path_to_node[-1]
            amount_node.total_amount = total_amount

        # 2) work backwards computing the expected cost at all nodes and
        #    choosing action at choice nodes where expected cost is lowest
        for depth in range(self._max_depth, 0, -1):
            for amount_node in self.amount_nodes(depth):
                # propagate expected payoff to parent chance node
                parent = self.parent(amount_node)
                parent.expected_amount += (
                    amount_node.probability * amount_node.total_amount
                )
            for decision_node in self.decision_nodes(depth):
                # select minimum cost from child chance nodes
                children = self.children(decision_node)
                decision_node.expected_amount = opt_func([
                    child.expected_amount for child in children
                ])
                if depth > 1:
                    parent = self.parent(decision_node)
                    parent.expected_amount += (
                        parent.p_failure * decision_node.expected_amount
                    )

    def amounts(self) -> DataFrame:

        results = []
        for amount_node in self.amount_nodes():
            path_to_node = list(all_simple_paths(
                self._graph, self._root_node, amount_node
            ))[0]
            chance_nodes = [c for c in path_to_node
                            if isinstance(c, ChanceNode)]
            if chance_nodes[-1].p_success != 1:
                continue
            result = {}
            for c, node in enumerate(chance_nodes):
                result[f'choice_{c + 1}'] = str(node.name)
                result[f'amount_{c + 1}'] = node.amount
                result[f'expected_amount_{c + 1}'] = node.expected_amount
            results.append(result)
        return DataFrame(results)
