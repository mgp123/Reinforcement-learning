import copy
import math

import torch
from tqdm import tqdm

from src.agent import Agent
from src.agent_observer import ReplayBuffer, RewardObserver
from src.learner import Learner
from src.stochastic_policy import DeterministicPolicy
from src.utilities import list_of_tuples_to_tuple_of_tensors, update_exponential_average, copy_model_and_optimizer


# TODO modify so as being able to copy q_model without using its base class. Using base class can be problemaic with
#  for example type(q_model) == Sequential. Sequential() is an empry model
class TD3(Learner):
    def __init__(self,
                 environment, discount_factor,
                 a_model, a_optimizer,
                 q_model, q_optimizer):

        super(TD3, self).__init__(environment, discount_factor)
        self.a_model, self.a_optimizer = a_model, a_optimizer,
        self.q_model_1, self.q_optimizer_1 = q_model, q_optimizer
        # randomize second q_model weights so as to be 2 different networks
        # this breaks any pretraining that q_model might have, so that should be considered when using algorithm
        self.q_model_2, self.q_optimizer_2 = copy_model_and_optimizer(self.q_model_1, self.q_optimizer_1)

        self.a_model_target = copy.deepcopy(self.a_model)
        self.q_model_1_target = copy.deepcopy(self.q_model_1)
        self.q_model_2_target = copy.deepcopy(self.q_model_2)

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
                     noise_bound=None,
                     buffer_size=math.inf
                     ):

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
                    exponential_average_factor,
                    noise_bound=noise_bound,
                    noise_distribution=gaussian_noise
                )

            # if episode ended, update progress
            if done:
                current_episode += 1
                pbar.update(1)

        pbar.close()
        return DeterministicPolicy(self.a_model), reward_observer.get_rewards()

    def experience_replay(self,
                          transitions,
                          exponential_average_factor,
                          noise_distribution,
                          noise_bound):

        state, action, reward, state_next, done = list_of_tuples_to_tuple_of_tensors(transitions)
        reward = torch.unsqueeze(reward, 1)
        done = torch.unsqueeze(done, 1)

        self.update_bellman_error(state, action, reward, done, state_next, noise_distribution, noise_bound)

        # as q_model changed, we should update our estimate for max action of q_model
        self.update_a_model(state)

        # updates target networks by adding to exponential average of previous weights:
        # w_target = w_model * epsilon + w_target * (1-epsilon)
        update_exponential_average(self.q_model_1_target, self.q_model_1, exponential_average_factor)
        update_exponential_average(self.q_model_2_target, self.q_model_2, exponential_average_factor)
        update_exponential_average(self.a_model_target, self.a_model, exponential_average_factor)

    def update_a_model(self, state):
        action = self.a_model(state)
        loss = - self.q_model_1(state, action)
        loss = loss.mean()
        self.a_optimizer.zero_grad()
        # backward will also propagate gradient to q_model but, as we zero grad it in next iteration, it is not an issue
        loss.backward()
        self.a_optimizer.step()

    def update_bellman_error(self, state, action, reward, done, state_next, noise_distribution, noise_bound):
        # action for target of q_model
        noise = noise_distribution.sample()
        if noise_bound is not None:
            noise = torch.clamp(noise, min=-noise_bound, max=noise_bound)
        action_target = self.a_model_target(state_next) + noise

        q_target = torch.min(
            self.q_model_1_target(state_next, action_target),
            self.q_model_2_target(state_next, action_target))
        y_q = reward + self.discount_factor * q_target * (1 - done)
        y_q = y_q.detach()

        # q_models gradient step
        for q_model, q_optimizer in \
                [(self.q_model_1, self.q_optimizer_1),
                 (self.q_model_2, self.q_optimizer_2)]:
            loss = torch.nn.MSELoss()
            loss = loss(q_model(state, action), y_q)
            loss = loss.mean()
            q_optimizer.zero_grad()
            loss.backward()
            q_optimizer.step()


