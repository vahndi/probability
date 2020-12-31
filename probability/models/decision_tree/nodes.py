from typing import Optional


class NodeMixin(object):
    """
    Base Node object with universal properties.
    """
    name: str
    depth: int


class DecisionNode(NodeMixin, object):
    """
    A Node representing a Decision with a number of Choices.
    """
    def __init__(
            self,
            name: str,
            depth: Optional[int] = None
    ):
        """
        Create a new DecisionNode.

        :param name: The name of the Decision.
        :param depth: The depth of the Node, starting at 1.
        """
        self.name: str = name
        self.depth: int = depth
        self.expected_amount: float = 0.0

    @property
    def str_amount(self) -> str:
        return f'{round(self.expected_amount, 2)}'


class ChanceNode(NodeMixin, object):
    """
    A Node representing a choice with a fixed chance of success and a fixed
    cost or reward (amount).
    """
    def __init__(
            self,
            name: str,
            p_success: float,
            amount: float,
            depth: Optional[int] = None
    ):
        """
        Create a new Chance Node.

        :param name: Name of the choice.
        :param p_success: Probability that the choice succeeds.
        :param amount: The amount (of cost or reward) of taking the choice.
        :param depth: The depth of the Node, starting at 1.
        """
        self.name: str = name
        self.p_success: float = p_success
        self.amount: float = amount
        self.depth: int = depth
        self.expected_amount: float = 0.0

    @property
    def p_failure(self) -> float:
        return 1 - self.p_success

    @property
    def str_amount(self) -> str:
        return f'{round(self.amount, 2)}, {round(self.expected_amount, 2)}'


class AmountNode(NodeMixin, object):
    """
    A Node representing a Cost or Reward if a choice is successful.
    """
    def __init__(
            self,
            name: str,
            probability: float,
            depth: Optional[int] = None
    ):
        """
        Create a new AmountNode.

        :param name: The name of the AmountNode.
        :param probability: The probability that the parent Choice was
                            successful.
        :param depth: The depth of the Node, starting at 1.
        """
        self.name: str = name
        self.depth: int = depth
        self.probability: float = probability
        self.total_amount: float = 0.0

    @property
    def str_amount(self) -> str:
        return f'{round(self.total_amount, 2)}'
