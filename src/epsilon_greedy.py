from random import random, randint

import torch

from src.policy import Policy
from src.type_definitions import StateType, FunctionApproximatorType


class GreedyQPolicy(Policy):
    def __init__(self, q_model: FunctionApproximatorType):
        """
        Policy that chooses action maximizing  the q function

        :param q_model: Q approximator function. Typically a neural network.
                Should support the () operation with torch arrays. This is though with pytorch in mind.
                Policy will behave epsilon greedy with respect to this q_model
        """
        self.q_model = q_model

    def __call__(self, state: StateType):
        # to torch
        x = torch.tensor([state], dtype=torch.float32)
        q = self.q_model(x)[0]

        # max Q action
        action = torch.argmax(q, dim=0).item()
        return action


class EpsilonGreedyQPolicy(Policy):
    def __init__(self, q_model: FunctionApproximatorType, epsilon: float):
        """
        Policy that chooses action maximizing the q function with epsilon chance of random action

        :param q_model: Q approximator function. Typically a neural network.
                Should support the () operation. This is though with pytorch in mind
                Policy will behave epsilon greedy with respect to this
        :param epsilon: probability of taking random action
        """
        super(EpsilonGreedyQPolicy, self).__init__()
        self.q_model = q_model
        self.epsilon = epsilon

    def __call__(self, state: StateType):
        # to torch
        x = torch.tensor([state], dtype=torch.float32)
        q = self.q_model(x)[0]

        action = torch.argmax(q, dim=0).item()

        if random() < self.epsilon:
            a_size = q.shape[0]
            action = randint(0, a_size-1)

        return action


class DecayingEpsilonGreedyQPolicy(EpsilonGreedyQPolicy):
    def __init__(self, q_model: FunctionApproximatorType, initial_epsilon: float, decay_factor: float, min_epsilon=0.1):
        """
        Policy that chooses action maximizing the q function with epsilon chance of random action.
        Epsilon decays after each episode

        :param q_model: Q approximator function. Typically a neural network.
                SShould support the () operation with torch arrays. This is though with pytorch in mind
                Policy will behave epsilon greedy with respect to this
        :param initial_epsilon: initial probability of taking random action
        :param decay_factor: multiplier to epsilon after end of each episode. Should be between 0 and 1
        """
        super(DecayingEpsilonGreedyQPolicy, self).__init__(q_model, initial_epsilon)
        self.decay_factor = decay_factor
        self.min_epsilon = min_epsilon

    def on_episode_end(self):
        self.epsilon *= self.decay_factor
        self.epsilon = max(self.epsilon, self.min_epsilon)
