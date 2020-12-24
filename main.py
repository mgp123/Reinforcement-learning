import gym
import torch
import torch.nn as nn

from agent import Agent
from epsilon_greedy import GreedyQPolicy, DecayingEpsilonGreedyQPolicy
from q_iteration import QIteration


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# TODO solve in a cleaner way and move to another file
class NetWithOptimizer(object):
    def __init__(self, module, optimizer, loss_fn):
        self.module = module
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def gradient_step(self, x, y):
        self.optimizer.zero_grad()
        output = self.module(x)
        loss = self.loss_fn(output, y)
        loss.backward()
        self.optimizer.step()

    def fit(self, x, y):
        return self.gradient_step(x, y)


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
        loss_fn = nn.MSELoss()
        q_model = NetWithOptimizer(q_model, optimizer, loss_fn)

        learner = QIteration(environment=environment, q_model=q_model)
        opt_policy = learner.learn_policy()

        torch.save(q_model.module, "learned networks/cartpole/q_network.dnet")

        agent = Agent(environment=environment, policy=opt_policy)
        input("add anything to continue")
        agent.perform_episode(render=True)

    else:
        environment = gym.make("CartPole-v1")
        q_model = torch.load("learned networks/cartpole/q_network.dnet")
        opt_policy = GreedyQPolicy(q_model)
        agent = Agent(environment=environment, policy=opt_policy)
        agent.perform_episode(render=True)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    environment = gym.make("MountainCar-v0")
    q_model = nn.Sequential(
        nn.Linear(2, 64),
        nn.ReLU(),
        nn.Linear(64, 3)
    )

    optimizer = torch.optim.Adam(q_model.parameters(), lr=0.002)
    loss_fn = nn.MSELoss()
    q_model = NetWithOptimizer(q_model, optimizer, loss_fn)

    exploration = DecayingEpsilonGreedyQPolicy(q_model, initial_epsilon=1.0, decay_factor=0.95, min_epsilon=0.05)
    learner = QIteration(environment=environment, q_model=q_model, exploration_policy=exploration)
    opt_policy = learner.learn_policy(episodes=100)

    agent = Agent(environment=environment, policy=opt_policy)
    input("add anything to continue")
    agent.perform_episode(render=True)
