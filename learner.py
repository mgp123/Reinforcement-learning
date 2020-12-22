from policy import Policy
from type_definitions import *


class Learner(object):
    def __init__(self, environment):
        self.environment = environment

    def learn_policy(self, *args, **kwargs) -> Policy:
        raise NotImplementedError
