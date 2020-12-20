import random

from policy import Policy
from type_definitions import *


class Learner(object):
    def __init__(self, environment):
        self.environment = environment
        self.sampled_trajectories = []

    def begin_trajectory(self, **kwargs):
        self.sampled_trajectories.append([])

    def add_transition(self, state=None, action=None, reward=None, **kwargs):
        current_trajectory_index = len(self.sampled_trajectories) - 1
        self.sampled_trajectories[current_trajectory_index] += [(state, action, reward)]

    def sample_trajectories(self, agent, episodes):
        """
        Runs complete episodes with agent and adds the trajectories to sampled_trajectories

        :param agent: the agent that should perform the episodes
        :param episodes: number of trajectories to store
        """
        for episode in range(episodes):
            agent.perform_episode(
                before_start_of_episode=[self.begin_trajectory],
                after_each_step=[self.add_transition]
            )

    def clear_trajectories(self):
        self.sampled_trajectories = []

    @staticmethod
    def get_reward_to_go(trajectory: TrajectoryType, discount_factor: float) -> List[float]:
        """
        :param discount_factor:
        :param trajectory: ordered list of transitions (state, action, reward)
        :return: the list of discounted reward to go for each point in trajectory
        """

        T = len(trajectory)
        reward_index = 2  # TODO get the index of reward in a better cleaner way
        reward_to_go = []
        current_reward_to_go = 0
        for t in range(T-1, -1, 0):
            reward = trajectory[t][reward_index]
            current_reward_to_go = reward + discount_factor*current_reward_to_go
            reward_to_go = [current_reward_to_go] + reward_to_go
        return reward_to_go

    def sample_transitions_from_stored_trajectories(self, n_samples) \
            -> List[Tuple[StateType, ActionType, float, StateType]]:
        """
        Samples transitions from stored trajectories. In case of terminal state, it gives none to state_next

        :param n_samples: number of samples to take
        :return: a list of (state, action, reward, state_next).
        """
        n = self.amount_of_stored_transitions()
        ind_sample = [random.randint(0, n) for _ in range(n_samples)]
        res = []
        state_index = 0

        # TODO make implementation faster by ordering ind_sample and going through sampled_trajectories once
        for i in ind_sample:
            for trajectory in self.sampled_trajectories:
                if i >= len(trajectory):
                    i -= len(trajectory)
                else:
                    state_next = None
                    if i < len(trajectory) - 1:
                        state_next = trajectory[i+1][state_index]

                    transition = trajectory[i] + (state_next,)
                    res.append(transition)
                    break
        return res

    def amount_of_stored_transitions(self) -> int:
        """
        Sums all the transitions in all the already sampled trajectories
        """
        res = 0
        for t in self.sampled_trajectories:
            res += len(t)
        return res

    def learn_policy(self, *args, **kwargs) -> Policy:
        raise NotImplementedError
