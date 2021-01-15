import torch
from tqdm import tqdm

from src.agent import Agent
from src.agent_observer import TrajectoryObserver, RewardObserver
from src.learner import Learner
from src.utilities import list_of_tuples_to_tuple_of_tensors, get_reward_to_go
from src.stochastic_policy import StochasticPolicy


class PolicyGradient(Learner):

    def __init__(self, environment, a_distribution_model, optimizer, discount_factor=0.95):
        """
        Performs vanilla policy gradient, that is, using reward to go
        """
        super(PolicyGradient, self).__init__(environment, discount_factor)
        self.a_distribution_model = a_distribution_model
        self.optimizer = optimizer

    def learn_policy(self, epochs=200, episodes_per_update=1):
        self.optimizer.zero_grad()
        state_index = 0
        action_index = 1

        policy = StochasticPolicy(self.a_distribution_model)
        agent = Agent(self.environment, policy)

        # utilities to collect agent data
        t_obs = TrajectoryObserver()
        r_obs = RewardObserver()
        agent.attach_observer(t_obs)
        agent.attach_observer(r_obs)

        for _ in tqdm(range(epochs)):
            for _ in range(episodes_per_update):
                # perform complete episode with observers attached
                agent.perform_episode()

                # collect trajectory and calculate reward to go
                trajectory = t_obs.last_trajectory()
                reward_to_go = get_reward_to_go(trajectory, self.discount_factor)

                # convert to pytorch tensors
                trajectory = list_of_tuples_to_tuple_of_tensors(trajectory)
                reward_to_go = torch.tensor(reward_to_go, dtype=torch.float32)

                # calculate loss
                policy_loss = self.a_distribution_model(trajectory[state_index])
                policy_loss = - policy_loss.log_prob(trajectory[action_index]) * reward_to_go
                policy_loss = torch.sum(policy_loss)

                # to estimate the expected gradient of episodes_per_update episodes,
                # we divide the loss by episodes_per_update
                policy_loss = policy_loss / episodes_per_update

                # accumulate gradient
                policy_loss.backward()

            # gradient step
            self.optimizer.step()
            self.optimizer.zero_grad()

            t_obs.clear()

        return policy, r_obs.get_rewards()
