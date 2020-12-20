from random import random, randint

import numpy as np

from policy import Policy
from type_definitions import StateType


class GreedyQPolicy(Policy):
    def __init__(self, q_model):
        """
        :param q_model: Q aproximator function. Typically a neural network.
                Should support the () operation. This is though with pytorch in mind
                Policy will behave epsilon greedy with respect to this
        """
        self.q_model = q_model

    def __call__(self, state: StateType):
        # to numpy
        x = np.asarray([state])
        q = self.q_model(x)

        # max Q action
        action = np.argmax(q, axis=0)
        return action


class EpsilonGreedyQPolicy(Policy):
    def __init__(self, q_model, epsilon: float):
        """
        :param q_model: Q aproximator function. Typically a neural network.
                Should support the () operation. This is though with pytorch in mind
                Policy will behave epsilon greedy with respect to this
        :param epsilon: probability of taking random action
        """
        super(EpsilonGreedyQPolicy, self).__init__()
        self.q_model = q_model
        self.epsilon = epsilon

    def __call__(self, state: StateType):
        # to numpy
        x = np.asarray([state])
        q = self.q_model(x)

        action = np.argmax(q, axis=0)

        if random() < self.epsilon:
            a_size = q.shape[1]
            action = randint(0, a_size-1)

        return action


class DecayingEpsilonGreedyQPolicy(EpsilonGreedyQPolicy):
    def __init__(self, initial_epsilon: float, q_model, decay_factor: float):
        """
        :param q_model: Q aproximator function. Typically a neural network.
                Should support the () operation. This is though with pytorch in mind
                Policy will behave epsilon greedy with respect to this
        :param initial_epsilon: initial probability of taking random action
        :param decay_factor: multiplier to epsilon after end of each episode. Should be between 0 and 1
        """
        super(DecayingEpsilonGreedyQPolicy, self).__init__(initial_epsilon, q_model)
        self.decay_factor = decay_factor

    def on_episode_end(self):
        self.epsilon *= self.decay_factor
