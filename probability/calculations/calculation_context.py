from probability.custom_types.calculation_types import CalculationValue


class CalculationContext(object):
    """
    Class to hold context of object in a calculation.
    """
    def __init__(self):
        """
        Create a new CalculationContext.
        """
        self._context = {}

    def __setitem__(self, name: str, value: CalculationValue):
        """
        Add an item to the Context.

        :param name: The name of the item.
        :param value: The item's value.
        """
        self._context[name] = value

    def __getitem__(self, name: str) -> CalculationValue:
        """
        Return a Context item.

        :param name: Name of the item to return.
        """
        return self._context[name]

    def context(self) -> dict:
        """
        Return the wrapped dictionary.
        """
        return self._context

    def has_object_named(self, name: str) -> bool:
        """
        Check if the CalculationContext contains an item.

        :param name: The name of the item to check for.
        """
        return name in self._context.keys()
