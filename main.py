from random import seed

import gym
import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal

from actor_critic import ActorCriticMonteCarloVEstimate
from agent import Agent
from epsilon_greedy import GreedyQPolicy, DecayingEpsilonGreedyQPolicy
from policy_gradient import PolicyGradient, StochasticPolicy
from q_iteration import QIteration


def cartpoloe():
    training = True

    if training:
        environment = gym.make("CartPole-v1")
        q_model = nn.Sequential(
            nn.Linear(4, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 2)
        )

        optimizer = torch.optim.Adam(q_model.parameters(), lr=0.001)

        learner = QIteration(
            environment=environment,
            q_model=q_model, optimizer=optimizer,
            exploration_policy=
            DecayingEpsilonGreedyQPolicy(q_model, initial_epsilon=1.0, decay_factor=0.95, min_epsilon=0.05)
        )

        opt_policy = learner.learn_policy(episodes=200)

        # torch.save(q_model.module, "learned networks/cartpole/q_network.torch")

        agent = Agent(environment=environment, policy=opt_policy)
        input("add anything to continue")
        agent.perform_episode(render=True)

    else:
        environment = gym.make("CartPole-v1")
        q_model = torch.load("learned networks/cartpole/q_network.torch")
        opt_policy = GreedyQPolicy(q_model)
        agent = Agent(environment=environment, policy=opt_policy)
        agent.perform_episode(render=True)


def pl_grad():
    print("hi")
    environment = gym.make("CartPole-v1")
    net = nn.Sequential(
        nn.Linear(4, 40, bias=False),
        nn.ReLU(),
        nn.Linear(40, 2, bias=False),
        nn.Softmax(dim=1)
    )

    class distributionNet(nn.Module):
        def __init__(self):
            super(distributionNet, self).__init__()
            self.net = net

        def forward(self, x):
            return Categorical(self.net(x))

    a_model = distributionNet()
    optimizer = torch.optim.Adam(a_model.parameters(), lr=0.01)
    learner = PolicyGradient(environment, a_model, optimizer, discount_factor=0.99)
    opt_policy = learner.learn_policy(epochs=500, episodes_per_update=1)

    agent = Agent(environment=environment, policy=opt_policy)
    input("add anything to continue")
    agent.perform_episode(render=True)

    # torch.save(a_distribution_model, "learned networks/cartpole/a_network.torch")


def walker():
    global action_distribution_model
    environment = gym.make("BipedalWalker-v3")
    print(environment.action_space)
    print(environment.observation_space)

    class action_distribution_model(nn.Module):

        def __init__(self):
            super(action_distribution_model, self).__init__()
            self.secquential = nn.Sequential(
                nn.Linear(24, 24),
                nn.ReLU(),
            )

            self.mean_layer = nn.Linear(24, 4)
            self.covariance_layer = nn.Linear(24, 4 ** 2)

        def forward(self, x):
            x = self.secquential(x)
            mean = self.mean_layer(x)

            covariance = self.covariance_layer(x)
            covariance = covariance.view(-1, 4, 4)
            covariance = torch.matmul(covariance, torch.transpose(covariance, 1, 2))  # semi-definite positive
            #  property: alpha * I + sigma is positive definite for alpha != 0 and sigma semi def pos
            # this also causes each component variance to have a minimum of alpha
            # so alpha should be chosen carefully as to not perturb optimal solution too much
            covariance = covariance + torch.eye(4) * 0.005

            return MultivariateNormal(mean, covariance)

    distribution = action_distribution_model()
    optimizer = torch.optim.Adam(distribution.parameters(), lr=0.01)
    learner = PolicyGradient(environment, distribution, optimizer)
    opt_policy = learner.learn_policy(100)
    a = Agent(environment, opt_policy)
    while input("continue ") == "c":
        a.perform_episode(render=True)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    seed(2020)
    torch.manual_seed(2020)

    environment = gym.make("CartPole-v1")
    net = nn.Sequential(
        nn.Linear(4, 40, bias=False),
        nn.ReLU(),
        nn.Linear(40, 2, bias=False),
        nn.Softmax(dim=1)
    )


    class distributionNet(nn.Module):
        def __init__(self):
            super(distributionNet, self).__init__()
            self.net = net

        def forward(self, x):
            return Categorical(self.net(x))


    a_model = distributionNet()
    optimizer = torch.optim.Adam(a_model.parameters(), lr=0.01)
    v_model = nn.Sequential(
        nn.Linear(4, 40, bias=False),
        nn.ReLU(),
        nn.Linear(40, 1, bias=False)
    )
    v_optimizer = torch.optim.Adam(v_model.parameters(), lr=0.01)

    learner = ActorCriticMonteCarloVEstimate(
        environment,
        a_model, optimizer,
        v_model, v_optimizer,
        discount_factor=0.99)

    opt_policy = learner.learn_policy(epochs=200, episodes_per_update=1)

    agent = Agent(environment=environment, policy=opt_policy)
    input("add anything to continue")
    agent.perform_episode(render=True)
