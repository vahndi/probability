from unittest.case import TestCase
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from probability.distributions.continuous.pert import PERT
from probability.distributions.data.ratio import Ratio
from probability.models.project_network.project_network import ProjectNetwork
from probability.models.project_network.project_task import ProjectTask


class TestProjectNetwork(TestCase):

    def setUp(self) -> None:
        self.net = ProjectNetwork()
        self.t1 = ProjectTask('A', PERT(1, 2, 3))
        self.t2 = ProjectTask('B', PERT(2, 4, 6))
        self.t3 = ProjectTask('C', PERT(1, 3, 5))

        self.net.add_task(self.t1)
        self.net.add_task(self.t2, parents='A', end=True)
        self.net.add_task(self.t3, parents='A', end=True)

    def test_project_network_paths_and_percentiles(self):
        self.assertIn('A', self.net.task_names)
        self.assertIn('B', self.net.task_names)
        self.assertIn('C', self.net.task_names)
        self.assertEqual(self.t1, self.net.task('A'))

        self.net.add_percentiles(0.95)
        df = self.net.path_lengths()
        self.assertEqual(2, len(df))
        self.assertTrue((df['length'] > 0).all())

    def test_simulation(self):
        sim_df = self.net.simulate(n_samples=500, random_state=42)
        self.assertEqual(len(sim_df), 500)
        self.assertIn('completion_time', sim_df.columns)
        self.assertIn('critical_path', sim_df.columns)
        self.assertIn('duration_A', sim_df.columns)
        self.assertIn('duration_B', sim_df.columns)
        self.assertIn('duration_C', sim_df.columns)

        # Completion time must be at least min(A) + min(B or C) >= 1 + 1 = 2
        self.assertTrue((sim_df['completion_time'] >= 2.0).all())

    def test_completion_time_distribution(self):
        dist = self.net.completion_time_distribution(n_samples=500, random_state=42)
        self.assertIsInstance(dist, Ratio)
        # Expected duration A ~ 2, B ~ 4, C ~ 3. Max path is through B (~6.0)
        mean_dur = dist.mean()
        self.assertGreater(mean_dur, 5.0)
        self.assertLess(mean_dur, 7.5)

        q95 = dist.quantile(0.95)
        self.assertGreater(q95, mean_dur)

    def test_critical_path_probabilities(self):
        probs = self.net.critical_path_probabilities(n_samples=500, random_state=42)
        self.assertEqual(len(probs), 2)
        self.assertAlmostEqual(probs.sum(), 1.0)

        # Path through B (mean 4) has higher probability than path through C (mean 3)
        path_b = ('START', 'A', 'B', 'END')
        path_c = ('START', 'A', 'C', 'END')
        self.assertIn(path_b, probs.index)
        self.assertIn(path_c, probs.index)
        self.assertGreater(probs[path_b], probs[path_c])

    def test_task_criticality_index(self):
        crit = self.net.task_criticality_index(n_samples=500, random_state=42)
        # Task A is on every simple path from START to END, so criticality must be 1.0
        self.assertAlmostEqual(crit['A'], 1.0)
        # Task B and Task C criticality must sum to 1.0
        self.assertAlmostEqual(crit['B'] + crit['C'], 1.0)
        self.assertGreater(crit['B'], crit['C'])

    def test_plot_completion_time(self):
        fig, ax = plt.subplots()
        ret_ax = self.net.plot_completion_time(n_samples=200, random_state=42, ax=ax)
        self.assertIs(ret_ax, ax)
        plt.close(fig)
