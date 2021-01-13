import torch

from src.policy import Policy
from src.type_definitions import StateType


class StochasticPolicy(Policy):

    def __init__(self, a_distribution_model):
        """
        Policy that chooses action according to the distribution of actions given by actor(state)

        :param a_distribution_model: Model that for any given state, outputs a torch.distribution of actions
        """
        self.a_distribution_model = a_distribution_model

    def __call__(self, state: StateType):
        x = torch.tensor([state], dtype=torch.float32)
        distribution = self.a_distribution_model(x)
        action = distribution.sample()[0]
        return action.tolist()


class DeterministicPolicy(Policy):

    def __init__(self, f, additive_noise_distribution=None):
        """
        Policy that chooses action by applying f(state)

        :param f: function determining the action to take
        :param additive_noise_distribution: optional, to be used as a distribution to sample to add to f(state).
        """

        self.f = f
        self.additive_noise_distribution = additive_noise_distribution

    def __call__(self, state: StateType):
        x = torch.tensor([state], dtype=torch.float32)
        y = self.f(x)[0]
        if self.additive_noise_distribution is not None:
            epsilon = self.additive_noise_distribution.sample()[0]
            y += epsilon

        return y.tolist()
