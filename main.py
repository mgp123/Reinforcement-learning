import gym
import torch
import torch.nn as nn

from agent import Agent
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
        self.fc1 = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc1(x)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    environment = gym.make("CartPole-v1")

    q_model = CartPoleQNet()
    optimizer = torch.optim.Adam(q_model.parameters(), lr=0.0005)
    loss_fn = nn.MSELoss()
    q_model = NetWithOptimizer(q_model, optimizer, loss_fn)

    learner = QIteration(environment=environment, q_model=q_model)
    opt_policy = learner.learn_policy()

    agent = Agent(environment=environment, policy=opt_policy)
    input("add anything to continue")
    agent.perform_episode(render=True)
