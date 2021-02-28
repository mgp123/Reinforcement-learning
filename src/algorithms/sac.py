import copy
import math

import torch
from tqdm import tqdm

from src.agent import Agent
from src.agent_observer import ReplayBuffer, RewardObserver
from src.learner import Learner
from src.policy import Policy
from src.stochastic_policy import StochasticPolicy
from src.type_definitions import StateType
from src.utilities import list_of_tuples_to_tuple_of_tensors, update_exponential_average, copy_model_and_optimizer


class SAC(Learner):
    def __init__(self,
                 environment, discount_factor,
                 a_distribution_model, a_optimizer,
                 q_model, q_optimizer):

        super(SAC, self).__init__(environment, discount_factor)
        self.a_distribution_model, self.a_optimizer = a_distribution_model, a_optimizer
        self.q_model_1, self.q_optimizer_1 = q_model, q_optimizer
        # randomize second q_model weights so as to be 2 different networks
        # this breaks any pretraining that q_model might have, so that should be considered when using this algorithm
        self.q_model_2, self.q_optimizer_2 = copy_model_and_optimizer(self.q_model_1, self.q_optimizer_1)

        self.a_model_target = copy.deepcopy(self.a_distribution_model)
        self.q_model_1_target = copy.deepcopy(self.q_model_1)
        self.q_model_2_target = copy.deepcopy(self.q_model_2)

    def learn_policy(self,
                     episodes=200,
                     experience_replay_samples=32,
                     exponential_average_factor=0.01,
                     entropy_coefficient=0,
                     buffer_size=math.inf,
                     updates_per_replay=1
                     ):

        pbar = tqdm(total=episodes)

        policy = StochasticPolicy(self.a_distribution_model)
        buffer = ReplayBuffer(buffer_size)

        reward_observer = RewardObserver()
        agent = Agent(self.environment, policy)
        agent.attach_observer(reward_observer)

        current_episode = 0

        while current_episode < episodes:
            # collect transition
            state, action, reward, state_next, done = agent.step()
            # add to buffer
            buffer.add_transition(state, action, reward, state_next, done)

            # if enough transitions collected perform experience replay algorithm
            if buffer.size() >= experience_replay_samples:
                for _ in range(updates_per_replay):
                    self.experience_replay(
                        buffer.sample_transitions(experience_replay_samples),
                        exponential_average_factor,
                        entropy_coefficient
                    )

            # if episode ended, update progress
            if done:
                current_episode += 1

                if current_episode % 20 == 0:
                    reward_observer.plot()
                    reward_observer.plot_moving_average(5)

                pbar.update(1)

        pbar.close()
        return MeanOfStochasticModel(self.a_distribution_model), reward_observer.get_rewards()

    def experience_replay(self,
                          transitions,
                          exponential_average_factor,
                          entropy_coefficient):

        state, action, reward, state_next, done = list_of_tuples_to_tuple_of_tensors(transitions)
        reward = torch.unsqueeze(reward, 1)
        done = torch.unsqueeze(done, 1)

        self.update_bellman_error(state, action, reward, done, state_next, entropy_coefficient)

        # as q_model changed, we should update our estimate for max action of q_model
        self.update_a_model(state, entropy_coefficient)

        # updates target networks by adding to exponential average of previous weights:
        # w_target = w_model * epsilon + w_target * (1-epsilon)
        update_exponential_average(self.q_model_1_target, self.q_model_1, exponential_average_factor)
        update_exponential_average(self.q_model_2_target, self.q_model_2, exponential_average_factor)

    def update_a_model(self, state, entropy_coefficient):
        action_distribution = self.a_distribution_model(state)
        action = action_distribution.sample()

        entropy = - entropy_coefficient * action_distribution.log_prob(action)

        loss = torch.min(
            self.q_model_1(state, action),
            self.q_model_2(state, action)
        ) + entropy

        loss = - loss.mean()

        self.a_optimizer.zero_grad()
        loss.backward()  # will also propagate to q_models but is zerod out in next iteration
        self.a_optimizer.step()

    def update_bellman_error(self, state, action, reward, done, state_next, entropy_coefficient):

        action_next_distribution = self.a_distribution_model(state_next)
        action_sampled = action_next_distribution.sample()
        entropy = - entropy_coefficient * action_next_distribution.log_prob(action_sampled)

        q_next = torch.min(
            self.q_model_1_target(state_next, action_sampled),
            self.q_model_2_target(state_next, action_sampled)
        ) + entropy

        y_q = reward + \
              self.discount_factor * q_next * (1 - done)

        y_q = y_q.detach()

        for q_model, q_optimizer in zip((self.q_model_1, self.q_model_2), (self.q_optimizer_1, self.q_optimizer_2)):
            # q_models gradient step
            loss = torch.nn.MSELoss()
            loss = loss(q_model(state, action), y_q)
            loss = loss.mean()
            q_optimizer.zero_grad()
            loss.backward()
            q_optimizer.step()


class MeanOfStochasticModel(StochasticPolicy):
    """
    Policy that chooses action according to the mean of a distribution of actions that depends on the state
    """

    def __call__(self, state: StateType):
        x = torch.tensor([state], dtype=torch.float32)
        distribution = self.a_distribution_model(x)
        action = distribution.mean[0]
        return action.tolist()
