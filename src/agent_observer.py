import math
from random import randint, shuffle, sample

from src.utilities import get_reward_to_go
from src.type_definitions import *
import matplotlib.pyplot as plt


class AgentObserver(object):
    def on_episode_start(self, **kwargs):
        raise NotImplementedError

    def on_step(self, **kwargs):
        raise NotImplementedError

    def on_episode_end(self):
        raise NotImplementedError


class TrajectoryObserver(AgentObserver):
    def __init__(self, buffer_size: int = math.inf):
        """
        Stores trajectories performed by agent it attaches to. Currently it only supports being attached to only one active agent
        :param buffer_size max amount of stored trajectories to have. Older trajectories are discarded
        """
        self.sampled_trajectories = []
        self.buffer_size = buffer_size

    def on_episode_start(self, **kwargs):
        if len(self.sampled_trajectories) == self.buffer_size:
            self.sampled_trajectories = self.sampled_trajectories[1:]

        self.sampled_trajectories.append([])

    def on_step(self, state=None, action=None, reward=None, **kwargs):
        current_trajectory_index = len(self.sampled_trajectories) - 1
        self.sampled_trajectories[current_trajectory_index] += [(state, action, reward)]

    def on_episode_end(self):
        pass

    def clear(self):
        self.sampled_trajectories = []

    def sample_transitions_from_stored_trajectories(self, n_samples) \
            -> List[Tuple[StateType, ActionType, float, StateType, bool]]:
        """
        Samples trajectories from stored trajectories. In case of terminal state, it loops to first in trajectory

        :param n_samples: number of samples to take
        :return: a list of (state, action, reward, state_next, done).
        """
        n = self.amount_of_stored_transitions()
        ind_samples = [randint(0, n - 1) for _ in range(n_samples)]
        ind_samples.sort()
        res = []
        state_index = 0

        iter_trajectory = iter(self.sampled_trajectories)
        current_transition_ind = 0
        trajectory = next(iter_trajectory)

        for ind_sample in ind_samples:
            # advance trajectories until sample is in current trajectory
            i = ind_sample - current_transition_ind
            while i >= len(trajectory):
                current_transition_ind += len(trajectory)
                trajectory = next(iter_trajectory)
                i = ind_sample - current_transition_ind

            # construct transition
            state_next = trajectory[0][state_index]  # in case there is no default state
            done = i == len(trajectory) - 1

            if not done:
                state_next = trajectory[i + 1][state_index]

            transition = trajectory[i] + (state_next, done)
            res.append(transition)

        # the trajectories are ordered in time, so they do not behave exactly like a batch of sampled trajectories
        # to solve this problem, shuffle res before return
        shuffle(res)

        return res

    def amount_of_stored_transitions(self) -> int:
        """
        Sums all the trajectories in all the already sampled trajectories
        """
        res = 0
        for t in self.sampled_trajectories:
            res += len(t)
        return res

    def get_trajectories(self):
        return self.sampled_trajectories

    def last_trajectory(self):
        return self.get_trajectories()[-1]

    def reward_to_go(self, discount_factor):
        return [get_reward_to_go(t, discount_factor) for t in self.sampled_trajectories]


class RewardObserver(AgentObserver):
    def __init__(self):
        self.trajectory_rewards = []

    def on_episode_start(self, **kwargs):
        self.trajectory_rewards.append(0)

    def on_step(self, reward=0, **kwargs):
        self.trajectory_rewards[-1] += reward

    def on_episode_end(self):
        pass

    def get_rewards(self):
        return self.trajectory_rewards

    def last_reward(self):
        return self.get_rewards()[-1]

    def plot(self):
        plt.plot(self.trajectory_rewards)
        plt.xlabel('episode')
        plt.ylabel('total reward')
        plt.savefig("score.png")

    def add(self, new_rewards):
        self.trajectory_rewards += new_rewards

    def size(self):
        return len(self.trajectory_rewards)


class ReplayBuffer(AgentObserver):

    def __init__(self, buffer_size=math.inf):
        self.transitions = []
        self.buffer_size = buffer_size

    def add_transition(self, state, action, reward, state_next, done):
        if self.size() == self.buffer_size:
            self.transitions = self.transitions[1:]
        self.transitions.append((state, action, reward, state_next, done))

    def size(self):
        return len(self.transitions)

    def sample_transitions(self, n_samples):
        return sample(self.transitions, n_samples)

    def on_episode_start(self, **kwargs):
        pass

    def on_step(self, state, action, reward, state_next, done):
        self.add_transition(state, action, reward, state_next, done)

    def on_episode_end(self):
        pass
