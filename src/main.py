from random import seed
from matplotlib import pyplot as plt


import gym
import torch
from torch import nn as nn

from src.agent import Agent
from src.algorithms.ddpg import DDPG


class Q_network(nn.Module):
    def __init__(self):
        super(Q_network, self).__init__()
        self.block = torch.nn.Sequential(
            nn.Linear(3 + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, state, action):
        x = torch.cat((state, action), 1)
        return self.block(x)


q_model = Q_network()
a_model = nn.Sequential(
    nn.Linear(3, 24),
    nn.ReLU(),
    nn.Linear(24, 1, bias=False),
    nn.Tanh()
)

lr = 0.01
a_opt = torch.optim.Adam(a_model.parameters(), lr=lr)
q_opt = torch.optim.Adam(q_model.parameters(), lr=lr)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    seed(2020)
    torch.manual_seed(2020)
    environment = gym.make("Pendulum-v0")
    learner = DDPG(
        environment, 0.99,
        a_model, a_opt,
        q_model, q_opt
    )

    policy, rew = learner.learn_policy(
        episodes=400,
        experience_replay_samples=32,
        gaussian_noise_variance=0.15,
        exponential_average_factor=0.005
    )

    plt.plot(rew)
    plt.xlabel('episode')
    plt.ylabel('total reward')
    plt.savefig("score.png")

    agent = Agent(environment, policy)
    while input("waiting") == "c":
        agent.perform_episode(render=True)



