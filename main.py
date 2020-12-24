import gym
import torch
import torch.nn as nn

from agent import Agent
from epsilon_greedy import GreedyQPolicy, DecayingEpsilonGreedyQPolicy
from policy_gradient import PolicyGradient
from q_iteration import QIteration


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


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

        #torch.save(q_model.module, "learned networks/cartpole/q_network.dnet")

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
    def pl_grad():
        print("hi")
        environment = gym.make("CartPole-v1")
        a_model = nn.Sequential(
            nn.Linear(4, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 2)
        )

        optimizer = torch.optim.Adam(a_model.parameters(), lr=0.005)

        learner = PolicyGradient(environment, a_model, optimizer)
        learner.learn_policy(epochs=1000, episodes_per_update=20)

    cartpoloe()

