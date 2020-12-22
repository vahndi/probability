from mpl_format.axes import AxesFormatter

from probability.models.project_network import ProjectNetwork, ProjectTask
from probability.distributions import PERT


def create_network():

    net = ProjectNetwork()
    net.add_task(ProjectTask('A', PERT(1, 4, 7)))
    net.add_task(ProjectTask('B1', PERT(2, 5, 8)), parents='A')
    net.add_task(ProjectTask('B2', PERT(3, 6, 9)), parents='A')
    net.add_task(ProjectTask('B3', PERT(2, 5, 14)), parents='A')
    net.add_task(ProjectTask('C', PERT(6, 21, 30)), parents='B1')
    net.add_task(ProjectTask('D1', PERT(5, 14, 17)), parents='B2')
    net.add_task(ProjectTask('D2', PERT(2, 11, 14)), parents='B2')
    net.add_task(ProjectTask('E', PERT(6, 21, 30)), parents='B3')
    net.add_task(ProjectTask('F', PERT(5, 8, 17)), parents=['C', 'D1'])
    net.add_task(ProjectTask('G', PERT(3, 9, 15)), parents=['D2', 'E'])
    net.add_task(ProjectTask('H', PERT(3, 12, 21)),
                 parents=['F', 'G'], end=True)
    return net


if __name__ == '__main__':

    network = create_network()
    axf = AxesFormatter()
    network.add_percentiles()
    network.draw(ax=axf.axes)
    lengths = network.path_lengths()
    axf.show()
