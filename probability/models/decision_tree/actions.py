from itertools import product as it_product
from typing import List, Optional, Callable, Iterator

from numpy import product as np_product

from probability.models.decision_tree.decision_tree import DecisionTree
from probability.models.decision_tree.nodes import ChanceNode, DecisionNode, \
    AmountNode


class Action(object):

    def __init__(self,
                 name: str,
                 p_success: float,
                 init_cost: float,
                 duration: float,
                 cost_inflation: float = 1):
        """
        An Action that could be tried.

        :param name: The name of the Action.
        :param p_success: The probability that the action is successful.
        :param init_cost: The cost of the action if it is tried straight away.
        :param duration: The duration of the action.
        :param cost_inflation: The exponential cost inflation per period.
        """
        self.name: str = name
        self.p_success: float = p_success
        self.init_cost: float = init_cost
        self.duration: float = duration
        self.cost_inflation: float = cost_inflation

    def cost(self, period: int) -> float:
        """
        Return the cost of taking the action starting at the given period.

        :param period: The period when the action is taken.
        """
        return self.init_cost * self.cost_inflation ** (period - 1)

    def __str__(self):

        return f'{self.name}'

    def __repr__(self):

        return f'{self.name}'


class AvailableAction(object):
    """
    A wrapper for an Action indicating whether it should be tried in the Trial.
    """
    def __init__(self, action: Action, try_action: bool):
        """
        Create a new ActionTrial.

        :param action: The Action.
        :param try_action: Whether the Action should be tried in the Trial.
        """
        self.action: Action = action
        self.try_action: bool = try_action

    def __str__(self):

        return f'{self.action.name} = {self.try_action}'


class ActionsTrial(object):
    """
    A Trial with a number of possible Actions,
    each of which may or may not be tried.
    """
    def __init__(self, *action_trials: AvailableAction):
        """
        Create a new ActionsTrial.

        :param action_trials: List of ActionTrial objects indicating whether
                              each Action should be tried.
        """
        self.action_trials: List[AvailableAction] = list(action_trials)

    def cost(self, period: int) -> float:
        """
        Return the total cost of trying all the Actions indicated as being
        tried.

        :param period: The period when the Trial is being run.
        """
        return sum([
            action_trial.action.cost(period=period)
            if action_trial.try_action is True
            else 0
            for action_trial in self.action_trials
        ])

    def p_success(self) -> float:
        """
        Return the total probability that at least one of the Actions tried is
        successful.
        """
        p_fails = [
            1 - action_trial.action.p_success
            if action_trial.try_action is True
            else 1
            for action_trial in self.action_trials
        ]
        return 1 - np_product(p_fails)

    def expected_cost_if_success(self, period: int) -> float:
        """
        Return the total expected cost of at least one of the Actions tried
        being successful.

        :param period:  The period when the Trial is being run.
        """
        expected_cost = 0
        for action_trial in self.action_trials:
            if action_trial.try_action:
                expected_cost += (
                        action_trial.action.p_success *
                        action_trial.action.cost(period=period)
                )
        return expected_cost

    def expected_cost_if_failure(self, period: int) -> float:
        """
        Return the total expected cost of none of the Actions tried
        being successful.

        :param period:  The period when the Trial is being run.
        """
        expected_cost = 0
        for action_trial in self.action_trials:
            if action_trial.try_action:
                expected_cost += (
                        (1 - action_trial.action.p_success) *
                        action_trial.action.cost(period=period)
                )
        return expected_cost

    def __str__(self):

        if all(action_trial.try_action is False
               for action_trial in self.action_trials):
            return 'No Action'
        else:
            return ', '.join([
                action_trial.action.name
                for action_trial in self.action_trials
                if action_trial.try_action is True
            ])

    def __repr__(self):

        args = ', '.join(
            [str(trial) for trial in self.action_trials]
        )
        return f'ActionsTrial({args})'


class ActionsGroup(object):
    """
    A group of Actions that can be used to generate Trials,
    where each action may or may not be tried.
    """
    def __init__(
            self, *actions: Action,
            can_try_action: Optional[Callable[[Action, int], bool]] = None,
            allow_no_action: bool = True,
            allow_multiple_actions: bool = True
    ):
        """
        Create a new ActionsGroup.

        :param actions: List of possible Actions available to try.
        """
        self.actions: List[Action] = list(actions)
        self.can_try_action: Optional[
            Callable[[Action, int], bool]
        ] = can_try_action
        self.allow_no_action: bool = allow_no_action
        self.allow_multiple_actions: bool = allow_multiple_actions

    @property
    def num_actions(self) -> int:
        """
        The number of Actions in the ActionsGroup.
        """
        return len(self.actions)

    def trials(self, period: int = 1) -> Iterator[ActionsTrial]:
        """
        Generate a sequence of ActionsTrial objects, representing all the
        different combinations of Action that could be tried, given the
        constraints of `allow_no_action` and `allow_multiple_actions`.

        :param period: The period in which the action is being tried.
        """
        for try_each_action in it_product(
                *[[False, True] for _ in range(self.num_actions)]
        ):
            # dont allow zero-action trials if not allowed
            if not self.allow_no_action and all([
                try_action is False for try_action in try_each_action
            ]):
                continue
            # don't allow multi-action trials if not allowed
            if not self.allow_multiple_actions and sum([
                try_action is True for try_action in try_each_action
            ]) > 1:
                continue

            if (
                    # all actions can be tried
                    self.can_try_action is None
                    # or all actions we want to try can be tried
                    or all(self.can_try_action(action, period) is True
                           or try_action is False
                           for action, try_action
                           in zip(self.actions, try_each_action))
            ):
                yield ActionsTrial(*(
                    AvailableAction(action=action, try_action=try_action)
                    for action, try_action
                    in zip(self.actions, try_each_action)
                ))
            else:
                continue

    def make_tree(self, max_depth: int) -> DecisionTree:

        tree = DecisionTree()

        num_decisions = 0
        num_amounts = 0

        depth = 1
        chance_nodes = [None]
        while depth <= max_depth:
            new_chance_nodes = []
            for chance_node in chance_nodes:
                if (
                        isinstance(chance_node, ChanceNode) and
                        chance_node.p_success == 1
                ):
                    continue
                # add decision node (and edge from chance node if there is one)
                num_decisions += 1
                decision_node = DecisionNode(
                    name=f'D{num_decisions}',
                    depth=depth
                )
                tree.add_decision_node(decision_node, chance_node)
                # add chance nodes and decision -> chance edges
                for choice in self.trials(period=depth):
                    if depth == max_depth and choice.p_success() < 1:
                        continue
                    chance_node = ChanceNode(
                        name=f'{choice}',
                        p_success=choice.p_success(),
                        amount=choice.cost(depth),
                        depth=depth,
                    )
                    tree.add_chance_node(chance_node, parent=decision_node)
                    new_chance_nodes.append(chance_node)
                    # add success node and chance -> success edge
                    num_amounts += 1
                    success_node = AmountNode(
                        name=f'P{num_amounts}',
                        probability=chance_node.p_success,
                        depth=depth
                    )
                    tree.add_amount_node(success_node, parent=chance_node)
                    # add failure node and chance -> failure edge if at max
                    # depth and success of choice is not guaranteed
                    if depth < max_depth or choice.p_success() == 1:
                        continue
                    num_amounts += 1
                    failure_node = AmountNode(
                        name=f'P{num_amounts}',
                        probability=chance_node.p_failure,
                        depth=depth,
                    )
                    tree.add_amount_node(failure_node, parent=chance_node)

            chance_nodes = new_chance_nodes
            depth += 1

        return tree
