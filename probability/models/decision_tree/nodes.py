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
            self, name: str, depth: int
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
    def str_amount(self):
        return f'{round(self.expected_amount, 2)}'


class ChanceNode(NodeMixin, object):
    """
    A Node representing a choice with a fixed chance of success and a fixed
    cost or reward (amount).
    """
    def __init__(
            self, name: str, depth: int,
            p_success: float, p_failure: float,
            amount: float,
    ):
        """
        Create a new Chance Node.

        :param name: Name of the choice.
        :param depth: The depth of the Node, starting at 1.
        :param p_success: Probability that the choice succeeds.
        :param p_failure: Probability that the choice fails.
        :param amount: The amount (of cost or reward) of taking the choice.
        """
        self.name: str = name
        self.depth: int = depth
        self.p_success: float = p_success
        self.p_failure: float = p_failure
        self.amount: float = amount
        self.expected_amount: float = 0.0

    @property
    def str_amount(self):
        return f'{round(self.amount, 2)}, {round(self.expected_amount, 2)}'


class AmountNode(NodeMixin, object):
    """
    A Node representing a Cost or Reward if a choice is successful.
    """
    def __init__(
            self, name: str, depth: int,
            probability: float,
    ):
        """
        Create a new AmountNode.

        :param name: The name of the AmountNode.
        :param depth: The depth of the Node, starting at 1.
        :param probability: The probability that the parent Choice was
                            successful.
        """
        self.name: str = name
        self.depth: int = depth
        self.probability: float = probability
        self.total_amount: float = 0.0

    @property
    def str_amount(self):
        return f'{round(self.total_amount, 2)}'
