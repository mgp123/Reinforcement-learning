import math
import numpy as np

from epsilon_greedy import DecayingEpsilonGreedyQPolicy, GreedyQPolicy
from learner import *
from agent import Agent
from policy import Policy


class QIteration(Learner):
    def __init__(self, environment, q_model, exploration_policy=None):
        """
            :param environment : should support step.
                This is the MDP that we wish to use reinforcement learning on
            :param q_model : Function approximation to use. Typically a neural network.
                Should support the () operation. This is though with pytorch in mind
            :param exploration_policy : policy to be used by agent to collect data for q iteration.
                Default is Decaying Epsilon Greedy using q_model with hyperparameters tuned
        """
        super(QIteration, self).__init__(environment)
        self.q_model = q_model
        self.discount_factor = 0.95

        self.hyperparameters = {
            # amount of samples to use to fit model in experience replay
            "experience_replay_samples": 32,
            # episodes to train learner
            "episodes_to_train": 50,
            # max amount of transitions the sampled trajectories should store
            "memory_buffer_size": math.inf,
            # how random should the policy be
            "epsilon_policy": 0.1,
            # decay factor for random should the policy
            "epsilon_decay_policy": 0.95
        }

        if exploration_policy is None:
            exploration_policy = DecayingEpsilonGreedyQPolicy(
                self.hyperparameters["epsilon_policy"],
                q_model,
                self.hyperparameters["epsilon_decay_policy"]
            )

        self.agent = Agent(
            environment,
            exploration_policy
        )

    def experience_replay(self, **kwargs):
        n_samples = self.hyperparameters["experience_replay_samples"]

        if n_samples < self.amount_of_stored_transitions():
            return

        transitions = self.sample_transitions_from_stored_trajectories(n_samples)
        transitions = np.asarray(transitions)

        state, action = transitions[:, 0], transitions[:, 1]
        reward, state_next = transitions[:, 2], transitions[:, 3]

        q_model = self.q_model

        # key algorithm part. Assuming finite possible states
        # There is a (unique) fixed point (fixed function?) Q* such that
        #   Q*(s_t,a_t) =  E [ r +  max_a  [Q*(s_t+1, a_t+1)] ]
        # This fixed point would be the optimal epsilon-greedy policy
        # in each experience replay call we approximate Q':
        #   Q' <- E [ r +  max_a  [Q(s_t+1, a_t+1)] ]   (sampling to approximate expectation)
        # where Q is our previous q_model

        # by using a function aproximator, there is, in theory no guaranteed convergence
        # Empirically however, it works

        q_max = np.amax(q_model(state_next), axis=1)
        target = q_model(state)
        target[np.arange(n_samples), action] = reward + self.discount_factor*q_max

        q_model.fit(state, target)

    def learn_policy(self) -> Policy:
        episodes = self.hyperparameters["episodes_to_train"]

        for episode in range(episodes):
            self.agent.perform_episode(
                before_start_of_episode=[self.begin_trajectory],
                after_each_step=[self.add_transition, self.experience_replay]
            )

        return GreedyQPolicy(q_model=self.q_model)
