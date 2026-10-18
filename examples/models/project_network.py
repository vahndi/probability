import matplotlib.pyplot as plt

from probability.models.project_network import ProjectNetwork, ProjectTask
from probability.distributions import PERT
from probability.plotting import new_axes


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
    ax = new_axes()
    network.add_percentiles()
    network.draw(ax=ax)
    lengths = network.path_lengths()
    print("Deterministic 95th-percentile path lengths:")
    print(lengths)

    print("\n--- Monte Carlo Project Analysis (2,000 runs) ---")
    crit_probs = network.critical_path_probabilities(n_samples=2000, random_state=42)
    print("Critical path probabilities:")
    print(crit_probs)

    print("\nTask criticality indices P(Task on critical path):")
    print(network.task_criticality_index(n_samples=2000, random_state=42))

    dist = network.completion_time_distribution(n_samples=2000, random_state=42)
    print(f"\nCompletion Time Distribution (Ratio): Mean={dist.mean():.2f}, Median={dist.median():.2f}, 95th Pct={dist.quantile(0.95):.2f}")

    fig, ax2 = plt.subplots(figsize=(8, 4))
    network.plot_completion_time(n_samples=2000, random_state=42, ax=ax2)
    plt.tight_layout()
    plt.show()
