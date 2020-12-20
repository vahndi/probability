"""
https://www.youtube.com/watch?v=KqgJSCtSnVE
https://www.youtube.com/watch?v=WtvrGE9vM2k
"""

from mpl_format.axes import AxesFormatter


# constants
from probability.models.decision_tree import Action, ActionsGroup, DecisionTree

MAX_YEAR = 2
ANNUAL_INFLATION = 1.2


def can_try(action: Action, year: int) -> bool:

    return year + action.duration <= 3


def create_actions_group() -> ActionsGroup:

    XNat = Action(
        name='X-NAT',
        p_success=2 / 3,
        init_cost=50,
        duration=0,
        cost_inflation=ANNUAL_INFLATION
    )
    XIvf = Action(
        name='X-IVF',
        p_success=1,
        init_cost=100,
        duration=0,
        cost_inflation=ANNUAL_INFLATION
    )
    MCap = Action(
        name='M-CAP',
        p_success=1 / 2,
        init_cost=10,
        duration=2,
        cost_inflation=ANNUAL_INFLATION
    )
    return ActionsGroup(XNat, XIvf, MCap,
                        can_try_action=can_try)


actions_group = create_actions_group()


def print_costs():

    print('\nCOSTS\n=====')
    for year in range(1, 4):
        print(f'year = {year}')
        for action in actions_group.actions:
            print(f'\t{action.name}.cost(year={year}) = {action.cost(year)}')


def print_can_trys():

    print('\nCAN TRY\n=======')
    for year in range(MAX_YEAR + 1):
        print(f'year = {year}')
        for action in actions_group.actions:
            print(f'\tcan_try({action.name}, {year}) = {can_try(action, year)}')


def print_trials():

    print('\nTRIALS\n======')
    for combo in actions_group.trials():
        print(f'{combo}')


def build_decision_tree() -> DecisionTree:

    dt = actions_group.make_tree(max_depth=3)
    return dt


def draw_tree(decision_tree: DecisionTree):

    axf = AxesFormatter(width=12, height=16)
    decision_tree.draw(node_labels='name', ax=axf.axes)
    axf.show()
    axf = AxesFormatter(width=12, height=16)
    decision_tree.draw(node_labels='amount', ax=axf.axes)
    axf.show()


if __name__ == '__main__':

    print_costs()
    print_can_trys()
    print_trials()
    tree = build_decision_tree()
    tree.solve()
    draw_tree(tree)
    costs = tree.amounts()
