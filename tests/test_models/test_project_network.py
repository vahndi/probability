from unittest.case import TestCase

from probability.distributions.continuous.pert import PERT
from probability.models.project_network.project_network import ProjectNetwork
from probability.models.project_network.project_task import ProjectTask


class TestProjectNetwork(TestCase):

    def test_project_network_paths_and_percentiles(self):

        net = ProjectNetwork()
        t1 = ProjectTask('A', PERT(1, 2, 3))
        t2 = ProjectTask('B', PERT(2, 4, 6))
        t3 = ProjectTask('C', PERT(1, 3, 5))

        net.add_task(t1)
        net.add_task(t2, parents='A', end=True)
        net.add_task(t3, parents='A', end=True)

        self.assertIn('A', net.task_names)
        self.assertIn('B', net.task_names)
        self.assertIn('C', net.task_names)
        self.assertEqual(t1, net.task('A'))

        net.add_percentiles(0.95)
        df = net.path_lengths()
        self.assertEqual(2, len(df))
        self.assertTrue((df['length'] > 0).all())
