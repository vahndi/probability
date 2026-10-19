from typing import Optional, List, Union, Dict
from matplotlib.axes import Axes
import numpy as np
from networkx import DiGraph, draw_networkx_nodes, draw_networkx_labels, \
    draw_networkx_edges, \
    shortest_path, draw_networkx_edge_labels, all_simple_paths
from pandas import DataFrame, Series

from probability.plotting import new_axes
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

    def simulate(
            self,
            n_samples: int = 1000,
            random_state: Optional[int] = None
    ) -> DataFrame:
        """
        Run Monte Carlo simulation of project execution.

        :param n_samples: Number of simulated project realizations.
        :param random_state: Optional random seed for reproducibility.
        :return: DataFrame with completion times, critical paths, and sampled task durations.
        """
        if random_state is not None:
            np.random.seed(random_state)

        paths = list(all_simple_paths(self._graph, self.task('START'), self.task('END')))
        if not paths:
            raise ValueError("No paths found from START to END in ProjectNetwork.")

        path_tuples = [tuple(t.name for t in p) for p in paths]

        # Draw duration samples for all tasks
        task_samples: Dict[str, np.ndarray] = {}
        for task in self.tasks:
            if task.name in ('START', 'END'):
                continue
            if task.duration is not None and hasattr(task.duration, 'rvs'):
                task_samples[task.name] = np.asarray(task.duration.rvs(n_samples), dtype=float)
            elif task.duration is not None and isinstance(task.duration, (int, float)):
                task_samples[task.name] = np.full(n_samples, float(task.duration))
            else:
                task_samples[task.name] = np.zeros(n_samples, dtype=float)

        # Compute durations for all paths across all samples
        path_matrix = np.zeros((n_samples, len(paths)), dtype=float)
        for i, p in enumerate(paths):
            p_tasks = [t for t in p if t.name not in ('START', 'END')]
            if p_tasks:
                path_matrix[:, i] = sum(task_samples[t.name] for t in p_tasks)

        # Completion time is max across all paths for each realization
        completion_times = np.max(path_matrix, axis=1)
        critical_path_idx = np.argmax(path_matrix, axis=1)
        critical_paths = [path_tuples[i] for i in critical_path_idx]

        results = {
            'completion_time': completion_times,
            'critical_path': critical_paths,
        }
        for task_name, samples in task_samples.items():
            results[f'duration_{task_name}'] = samples

        return DataFrame(results)

    def completion_time_distribution(
            self,
            n_samples: int = 1000,
            random_state: Optional[int] = None
    ):
        """
        Compute the Monte Carlo project completion-time distribution as a Ratio data distribution.

        :param n_samples: Number of simulation trials.
        :param random_state: Optional random seed.
        :return: Ratio distribution of project completion times.
        """
        from probability.distributions.data.ratio import Ratio
        sim_df = self.simulate(n_samples=n_samples, random_state=random_state)
        return Ratio(Series(sim_df['completion_time'], name='completion_time'))

    def critical_path_probabilities(
            self,
            n_samples: int = 1000,
            random_state: Optional[int] = None
    ) -> Series:
        """
        Compute the probability that each simple path from START to END is the critical path.

        :param n_samples: Number of simulation trials.
        :param random_state: Optional random seed.
        :return: Series of probabilities indexed by path tuples.
        """
        sim_df = self.simulate(n_samples=n_samples, random_state=random_state)
        paths = list(all_simple_paths(self._graph, self.task('START'), self.task('END')))
        path_tuples = [tuple(t.name for t in p) for p in paths]
        counts = sim_df['critical_path'].value_counts()
        probs = Series(0.0, index=path_tuples, name='critical_path_probability')
        for p, count in counts.items():
            probs[p] = count / n_samples
        return probs.sort_values(ascending=False)

    def task_criticality_index(
            self,
            n_samples: int = 1000,
            random_state: Optional[int] = None
    ) -> Series:
        """
        Compute the criticality index (probability of lying on the critical path) for each task.

        :param n_samples: Number of simulation trials.
        :param random_state: Optional random seed.
        :return: Series indexed by task name with criticality index in [0, 1].
        """
        sim_df = self.simulate(n_samples=n_samples, random_state=random_state)
        real_tasks = [t.name for t in self.tasks if t.name not in ('START', 'END')]
        crit_paths = sim_df['critical_path'].values

        crit_index = {}
        for t_name in real_tasks:
            count = sum(1 for cp in crit_paths if t_name in cp)
            crit_index[t_name] = count / n_samples

        return Series(crit_index, name='criticality_index').sort_values(ascending=False)

    def plot_completion_time(
            self,
            n_samples: int = 1000,
            random_state: Optional[int] = None,
            ax: Optional[Axes] = None,
            **kwargs
    ) -> Axes:
        """
        Plot the completion time distribution with mean and 95th percentile markers.
        """
        ratio = self.completion_time_distribution(n_samples=n_samples, random_state=random_state)
        ax = ratio.plot_kde(ax=ax, **kwargs)
        mean_val = ratio.mean()
        q95_val = ratio.quantile(0.95)
        ax.axvline(mean_val, color='darkblue', linestyle='--', label=f'Mean ({mean_val:.2f})')
        ax.axvline(q95_val, color='crimson', linestyle=':', label=f'95th Pct ({q95_val:.2f})')
        ax.set_title('Project Completion Time Distribution (Monte Carlo)')
        ax.set_xlabel('Duration')
        ax.set_ylabel('Density')
        ax.legend()
        return ax

    def draw(self, ax: Optional[Axes] = None) -> Axes:
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
        draw_networkx_edge_labels(G=self._graph, pos=pos, ax=ax,
                                  edge_labels=self._edge_weight_dict())
        return ax

