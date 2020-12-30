from policy import Policy
from type_definitions import *


class Learner(object):
    def __init__(self, environment, discount_factor):
        self.environment = environment
        self.discount_factor = discount_factor

    def learn_policy(self, *args, **kwargs) -> Tuple[Policy, List[float]]:
        raise NotImplementedError
