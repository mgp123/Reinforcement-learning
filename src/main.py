from random import seed
from matplotlib import pyplot as plt

import gym
import torch
from torch import nn as nn

from src.agent import Agent
from src.algorithms.ddpg import DDPG
from src.algorithms.sac import SAC
from src.algorithms.td3 import TD3


class Q_network(nn.Module):
    def __init__(self):
        super(Q_network, self).__init__()
        self.block = torch.nn.Sequential(
            nn.Linear(3 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action):
        x = torch.cat((state, action), 1)
        return self.block(x)


class A_Deterministic_network(nn.Module):
    def __init__(self):
        super(A_Deterministic_network, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(3, 24),
            nn.ReLU(),
            nn.Linear(24, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, state):
        return self.block(state)


v_model = nn.Sequential(
    nn.Linear(3, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)


class A_stothastic_network(nn.Module):
    def __init__(self):
        super(A_stothastic_network, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU()
        )
        self.mean_block = nn.Linear(64, 1)
        self.variance_block = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, state):
        x = self.block(state)

        mean = self.mean_block(x)
        variance = self.variance_block(x)

        y = torch.distributions.Normal(mean, variance)
        return y


def using_sac(environment):
    q_model = Q_network()
    a_model = A_stothastic_network()

    lr = 0.03
    a_opt = torch.optim.Adam(a_model.parameters(), lr=lr)
    q_opt = torch.optim.Adam(q_model.parameters(), lr=lr)
    v_opt = torch.optim.Adam(v_model.parameters(), lr=lr)

    learner = SAC(environment, 0.99, a_model, a_opt, q_model, q_opt, v_model, v_opt)

    policy, rew = learner.learn_policy(
        episodes=400,
        experience_replay_samples=128,
        exponential_average_factor=0.005,
        entropy_coefficient=0.002,
        buffer_size=100000
    )

    return policy, rew


def using_td3(environment):
    q_model = Q_network()
    a_model = nn.Sequential(
        nn.Linear(3, 24),
        nn.ReLU(),
        nn.Linear(24, 1, bias=False),
        nn.Tanh()
    )

    lr = 0.03
    a_opt = torch.optim.Adam(a_model.parameters(), lr=lr)
    q_opt = torch.optim.Adam(q_model.parameters(), lr=lr)

    learner = TD3(
        environment, 0.99,
        a_model, a_opt,
        q_model, q_opt
    )
    policy, rew = learner.learn_policy(
        episodes=300,
        experience_replay_samples=32,
        gaussian_noise_variance=0.15,
        noise_bound=0.15,
        exponential_average_factor=0.005
    )
    return policy, rew


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    seed(2020)
    torch.manual_seed(2020)
    environment = gym.make("Pendulum-v0")
    policy, rew = using_sac(environment)

    plt.plot(rew)
    plt.xlabel('episode')
    plt.ylabel('total reward')
    plt.savefig("score.png")

    agent = Agent(environment, policy)
    while input("waiting") == "c":
        agent.perform_episode(render=True)
