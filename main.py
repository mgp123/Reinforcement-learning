import gym
import torch
import torch.nn as nn

from agent import Agent
from epsilon_greedy import GreedyQPolicy
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


class CartPoleQNet(nn.Module):
    def __init__(self):
        super(CartPoleQNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 2)
        )

    def forward(self, x):
        return self.model(x)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    training = True

    if training:
        environment = gym.make("CartPole-v1")
        q_model = CartPoleQNet()
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



