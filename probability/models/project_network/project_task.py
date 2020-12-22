from typing import Optional

from probability.distributions.mixins.rv_mixins import ISFContinuous1dMixin


class ProjectTask(object):
    """
    A Task that will live in a ProjectNetwork.
    """
    def __init__(self, name: str,
                 duration: Optional[ISFContinuous1dMixin] = None):
        """
        Create a new ProjectTask.

        :param name: The name of the ProjectTask
        :param duration: A Continuous Probability Distribution representing
                             the duration of the Task.
        """
        self.name = name
        self.duration: Optional[ISFContinuous1dMixin] = duration
