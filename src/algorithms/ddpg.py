import copy
import math

import torch
from tqdm import tqdm

from src.agent import Agent
from src.agent_observer import ReplayBuffer, RewardObserver
from src.learner import Learner
from src.utilities import list_of_tuples_to_tuple_of_tensors, update_exponential_average
from src.stochastic_policy import DeterministicPolicy


class DDPG(Learner):
    def __init__(self,
                 environment, discount_factor,
                 a_model, a_optimizer,
                 q_model, q_optimizer):

        super(DDPG, self).__init__(environment, discount_factor)
        self.a_model, self.a_optimizer = a_model, a_optimizer,
        self.q_model, self.q_optimizer = q_model, q_optimizer

        self.a_model_target = copy.deepcopy(self.a_model)
        self.q_model_target = copy.deepcopy(self.q_model)

    def gaussian_distribution(self, gaussian_noise_variance):
        action_space_dimensions = self.environment.action_space.shape[0]
        gaussian_noise = torch.distributions.Normal(
            torch.zeros(action_space_dimensions),
            gaussian_noise_variance * torch.ones(action_space_dimensions)
        )
        return gaussian_noise

    def learn_policy(self,
                     episodes=200,
                     experience_replay_samples=32,
                     gaussian_noise_variance=1,
                     exponential_average_factor=0.01,
                     buffer_size=math.inf):

        pbar = tqdm(total=episodes)

        gaussian_noise = self.gaussian_distribution(gaussian_noise_variance)

        policy = DeterministicPolicy(
            self.a_model,
            additive_noise_distribution=gaussian_noise
        )
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
                self.experience_replay(
                    buffer.sample_transitions(experience_replay_samples),
                    exponential_average_factor)

            # if episode ended, update progress
            if done:
                current_episode += 1
                pbar.update(1)

        pbar.close()
        return DeterministicPolicy(self.a_model), reward_observer.get_rewards()

    def experience_replay(self, transitions, exponential_average_factor):
        state, action, reward, state_next, done = list_of_tuples_to_tuple_of_tensors(transitions)
        reward = torch.unsqueeze(reward, 1)
        done = torch.unsqueeze(done, 1)

        self.update_bellman_error(state, action, reward, done, state_next)

        # as q_model changed, we should update our estimate for max action of q_model
        self.update_a_model(state)

        # updates target networks by adding to exponential average of previous weights:
        # w_target = w_model * epsilon + w_target * (1-epsilon)
        update_exponential_average(self.q_model_target, self.q_model, exponential_average_factor)
        update_exponential_average(self.a_model_target, self.a_model, exponential_average_factor)

    def update_a_model(self, state):
        action = self.a_model(state)
        loss = - self.q_model(state, action)
        loss = loss.mean()
        self.a_optimizer.zero_grad()
        # backward will also propagate gradient to q_model but, as we zero grad it in next iteration, it is not an issue
        loss.backward()
        self.a_optimizer.step()

    def update_bellman_error(self, state, action, reward, done, state_next):
        # target for q_model
        action_target = self.a_model_target(state_next)
        y_q = reward + self.discount_factor * self.q_model_target(state_next, action_target) * (1 - done)
        y_q = y_q.detach()

        # q_model gradient step
        loss = torch.nn.MSELoss()
        loss = loss(self.q_model(state, action), y_q)
        loss = loss.mean()
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()
