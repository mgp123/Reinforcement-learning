import copy
import math

import torch
from tqdm import tqdm

from src.agent import Agent
from src.agent_observer import ReplayBuffer, RewardObserver
from src.learner import Learner
from src.stochastic_policy import StochasticPolicy
from src.utilities import list_of_tuples_to_tuple_of_tensors, update_exponential_average, copy_model_and_optimizer


class SAC(Learner):
    def __init__(self,
                 environment, discount_factor,
                 a_distribution_model, a_optimizer,
                 q_model, q_optimizer,
                 v_model, v_optimizer):

        super(SAC, self).__init__(environment, discount_factor)
        self.a_distribution_model, self.a_optimizer = a_distribution_model, a_optimizer,
        self.q_model, self.q_optimizer = q_model, q_optimizer

        self.v_model, self.v_optimizer = v_model, v_optimizer
        self.v_model_target = copy.deepcopy(self.v_model)

    def learn_policy(self,
                     episodes=200,
                     experience_replay_samples=32,
                     exponential_average_factor=0.01,
                     entropy_coefficient=0,
                     buffer_size=math.inf
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

                pbar.update(1)

        pbar.close()
        return policy, reward_observer.get_rewards()

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
        update_exponential_average(self.v_model_target, self.v_model, exponential_average_factor)

    def update_a_model(self, state, entropy_coefficient):
        action_distribution = self.a_distribution_model(state)
        action = action_distribution.sample()

        loss = self.q_model(state, action) - entropy_coefficient * action_distribution.log_prob(action)
        loss = - loss
        loss = loss.mean()
        self.a_optimizer.zero_grad()
        # backward will also propagate gradient to q_model but, as we zero grad it in next iteration, it is not an issue
        loss.backward()
        self.a_optimizer.step()

    def update_bellman_error(self, state, action, reward, done, state_next, entropy_coefficient):
        v_next = self.v_model_target(state_next)

        y_q = reward + self.discount_factor * v_next * (1 - done)

        y_q = y_q.detach()

        # q_models gradient step
        loss = torch.nn.MSELoss()
        loss = loss(self.q_model(state, action), y_q)
        loss = loss.mean()
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

        # action for target of v_model
        action_distribution = self.a_distribution_model(state)
        action_sampled = action_distribution.sample()

        y_v = (1 - done) * (
              self.q_model(state, action_sampled)
              - entropy_coefficient * action_distribution.log_prob(action_sampled)
        )

        y_v = y_v.detach()

        loss = torch.nn.MSELoss()
        loss = loss(self.v_model(state), y_v)
        loss = loss.mean()
        self.v_optimizer.zero_grad()
        loss.backward()
        self.v_optimizer.step()
