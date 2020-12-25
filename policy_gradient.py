import torch
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from tqdm import tqdm

from agent import Agent
from agent_observer import TrajectoryObserver, RewardObserver
from learner import Learner
from policy import Policy
from pytorch_utilities import list_of_tuples_to_tuple_of_tensors, get_reward_to_go
from type_definitions import StateType


class StochasticPolicy(Policy):

    def __init__(self, a_model):
        """
        Policy that chooses action according to the distribution of actions given by a_model(state)
        :param a_model: Model that for any given state, outputs a probabilistic distribution of actions
        """
        self.a_model = a_model

    def __call__(self, state: StateType):
        x = torch.tensor([state], dtype=torch.float32)
        distribution = self.a_model(x)
        # TODO Check if correct
        m = Categorical(distribution)
        action = m.sample().item()
        return action


class PolicyGradient(Learner):

    def __init__(self, environment, a_model, optimizer, discount_factor=0.95):
        super(PolicyGradient, self).__init__(environment, discount_factor)
        self.a_model = a_model
        self.optimizer = optimizer

    def learn_policy(self, epochs=200, episodes_per_update=1) -> Policy:
        self.optimizer.zero_grad()
        state_index = 0
        action_index = 1
        reward_history = []

        policy = StochasticPolicy(self.a_model)

        for _ in tqdm(range(epochs)):
            for _ in range(episodes_per_update):
                # collect trajectory and calculate reward to go
                trajectory, trajectory_reward = self.collect_trajectory(policy)
                reward_history.append(trajectory_reward)
                advantage = get_reward_to_go(trajectory, self.discount_factor)

                # convert to pytorch tensors
                trajectory = list_of_tuples_to_tuple_of_tensors(trajectory)
                advantage = torch.tensor(advantage, dtype=torch.float32)

                # calculate loss
                policy_loss = Categorical(self.a_model(trajectory[state_index]))
                policy_loss = - policy_loss.log_prob(trajectory[action_index]) * advantage
                policy_loss = torch.sum(policy_loss)

                # to take the expected gradient of episodes_per_update episodes,
                # we divide the loss by episodes_per_update
                policy_loss = policy_loss/episodes_per_update

                # accumulate gradient
                policy_loss.backward()

            # gradient step
            self.optimizer.step()
            self.optimizer.zero_grad()

        plt.plot(reward_history)
        plt.xlabel('episode')
        plt.ylabel('total reward')
        plt.savefig("score.png")

        return policy

    def collect_trajectory(self, policy):
        t_obs = TrajectoryObserver()
        r_obs = RewardObserver()

        agent = Agent(self.environment, policy)
        agent.attach_observer(t_obs)
        agent.attach_observer(r_obs)

        agent.perform_episode()

        return t_obs.get_trajectories()[0], r_obs.get_rewards()[0]
