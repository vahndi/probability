from typing import Optional, List, Union

from matplotlib.axes import Axes
from mpl_format.axes.axis_utils import new_axes
from networkx import DiGraph, draw_networkx_nodes, draw_networkx_labels, \
    draw_networkx_edges, \
    shortest_path, draw_networkx_edge_labels, all_simple_paths
from pandas import DataFrame

from probability.models.project_network.project_task import ProjectTask
from probability.models.utils import distribute_about_center


class ProjectNetwork(object):
    """
    A ProjectNetwork to estimate the duration of several dependent tasks.
    """
    def __init__(self):
        """
        Create a new ProjectNetwork.
        """
        self._graph: DiGraph = DiGraph()
        self._graph.add_node(ProjectTask('START'))
        self._graph.add_node(ProjectTask('END'))

    @property
    def _task_name_dict(self) -> dict:
        """
        Return a dict mapping tasks to task names.
        """
        return {
            node: node.name
            for node in self._graph.nodes()
        }

    def _edge_weight_dict(self) -> dict:
        """
        Return a dict mapping edges to their weights.
        """
        labels = {}
        for edge in self._graph.edges(data=True):
            if 'weight' in edge[2].keys():
                labels[(edge[0], edge[1])] = round(edge[2]['weight'], 3)
            else:
                labels[(edge[0], edge[1])] = ''
        return labels

    def _get_layout(self) -> dict:
        """
        Calculate the layout for drawing the network.
        """
        task_distances = {
            task: len(shortest_path(self._graph, self.task('START'), task)) - 1
            for task in self._graph.nodes()
        }
        max_distance = max([distance for distance in task_distances.values()])
        x = {
            task: distance / max_distance
            for task, distance in task_distances.items()
        }
        num_at_distance = {
            distance: sum([d == distance for d in task_distances.values()])
            for distance in range(max_distance + 1)
        }
        max_num_at_distance = max(num_at_distance.values())
        y = {}
        for distance in range(max_distance + 1):
            distance_tasks = {k: v for k, v in task_distances.items()
                                 if v == distance}
            for t, (task, task_distance) in enumerate(distance_tasks.items()):
                y[task] = distribute_about_center(
                    index=t,
                    size=num_at_distance[task_distance],
                    max_size=max_num_at_distance
                )
        return {
            task: [x[task], y[task]]
            for task in task_distances.keys()
        }

    def task(self, name: str) -> ProjectTask:
        """
        Return a task with a given name.

        :param name: The name of the ProjectTask.
        """
        if name not in self.task_names:
            raise ValueError(f'No Tasks named {name}.')
        return [node for node in self._graph.nodes()
                if node.name == name][0]

    @property
    def task_names(self) -> List[str]:
        """
        Return a list of names of each Task in the Network.
        """
        return [node.name for node in self._graph.nodes()]

    @property
    def tasks(self) -> List[ProjectTask]:
        """
        Return a list of Tasks in the Network.
        """
        return [node for node in self._graph.nodes()]

    def add_task(self, task: ProjectTask,
                 parents: Optional[Union[str, List[str]]] = None,
                 end: bool = False):
        """
        Add a new Task to the Network.

        :param task: A ProjectTask object.
        :param parents: The name(s) of the Tasks the ProjectTask depends on.
        :param end: Set to True if this Task has no following ProjectTask.
        """
        existing_names = self.task_names
        if task.name in existing_names:
            raise ValueError(f'Task named {task.name} already exists.')

        if parents is None:
            parents = ['START']
        elif isinstance(parents, str):
            parents = [parents]

        self._graph.add_node(task)
        for parent in parents:
            parent_node = self.task(parent)
            self._graph.add_edge(parent_node, task)
        if end:
            self._graph.add_edge(task, self.task('END'))

    def add_percentiles(self, value: float = 0.95):
        """
        Add the given percentile to each ProjectTask's out-edge.

        :param value: The value of the percentile from 0.0 to 1.0.
        """
        for task in self.tasks:
            for edge in self._graph.out_edges([task]):
                if task.duration is not None:
                    self._graph[edge[0]][edge[1]]['weight'] = (
                            task.duration.isf().at(1 - value)
                    )

    def path_lengths(self) -> DataFrame:
        """
        Return a DataFrame with columns of 'path' and 'length'.
        """
        paths = []
        for path in all_simple_paths(
                self._graph, self.task('START'), self.task('END')
        ):
            path_length = 0.0
            for n in range(len(path) - 1):
                if 'weight' in self._graph[path[n]][path[n + 1]].keys():
                    path_length += self._graph[path[n]][path[n + 1]]['weight']
            paths.append({
                'path': tuple(node.name for node in path),
                'length': path_length
            })
        return DataFrame(paths)

    def draw(self, ax: Optional[Axes]) -> Axes:
        """
        Draw the network.

        :param ax: Optional matplotlib Axes.
        """
        ax = ax or new_axes()
        pos = self._get_layout()
        draw_networkx_nodes(
            G=self._graph, pos=pos, ax=ax,
            nodelist=self._graph.nodes(),
            alpha=0.5,
        )
        draw_networkx_labels(
            G=self._graph, pos=pos, ax=ax,
            labels=self._task_name_dict, font_size=10,
        )
        draw_networkx_edges(G=self._graph, pos=pos, label='weight',
                            ax=ax, edge_color='gray')
        draw_networkx_edge_labels(G=self._graph, pos=pos,
                                  edge_labels=self._edge_weight_dict())
        return ax

