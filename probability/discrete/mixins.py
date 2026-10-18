from typing import Dict


class StatesMixin(object):

    _states: Dict[str, list]

    @property
    def states(self) -> Dict[str, list]:

        return self._states
