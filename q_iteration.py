import math
import torch

from agent_observer import TrajectoryObserver, RewardObserver
from epsilon_greedy import DecayingEpsilonGreedyQPolicy, GreedyQPolicy
from learner import *
from agent import Agent
from policy import Policy


class QIteration(Learner):
    def __init__(self, environment, q_model: FunctionApproximatorType, exploration_policy: Policy = None):
        """
        Performs Q iteration for environment using q_model and exploration_policy.
        q_model is updated

        :param environment :
            This is the MDP that we wish to use reinforcement learning on.
            Should support step.
        :param q_model : Function approximation to use. Typically a neural network.
            Should support the () operation with torch arrays and the fit operation. This is though with pytorch in mind.
        :param exploration_policy : Policy to be used by agent to collect data for q iteration.
            Default is Decaying Epsilon Greedy using q_model with hyperparameters tuned
        """
        super(QIteration, self).__init__(environment)
        self.q_model = q_model
        self.discount_factor = 0.99

        self.hyperparameters = {
            # amount of samples to use to fit model in experience replay
            "experience_replay_samples": 32,
            # episodes to train learner
            "episodes_to_train": 100,
            # max amount of transitions the sampled trajectories should store
            # TODO use buffer size
            "memory_buffer_size": math.inf,
            # how random should the policy be
            "epsilon_policy": 0.5,
            # decay factor for random should the policy
            "epsilon_decay_policy": 0.95
        }

        if exploration_policy is None:
            exploration_policy = DecayingEpsilonGreedyQPolicy(
                self.hyperparameters["epsilon_policy"],
                q_model,
                self.hyperparameters["epsilon_decay_policy"]
            )

        self.trajectories = TrajectoryObserver()
        self.exploration_policy = exploration_policy

    def experience_replay(self, **kwargs):
        n_samples = self.hyperparameters["experience_replay_samples"]

        if self.trajectories.amount_of_stored_transitions() < n_samples:
            return

        transitions = self.trajectories.sample_transitions_from_stored_trajectories(n_samples)

        # TODO transform to torch in a cleaner way
        state, action, reward, state_next, done = [], [], [], [], []
        for t in transitions:
            state.append(t[0])
            action.append(t[1])
            reward.append(t[2])
            state_next.append(t[3])
            done.append(t[4])

        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64)
        reward = torch.tensor(reward, dtype=torch.float32)
        state_next = torch.tensor(state_next, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        q_model = self.q_model

        # key algorithm part. Assuming finite possible states
        # There is a (unique) fixed point (fixed function?) Q* such that
        #   Q*(s_t,a_t) =  E [ r +  max_a  [Q*(s_t+1, a_t+1)] ]
        # This fixed point would be the optimal epsilon-greedy policy
        # in each experience replay call we approximate Q':
        #   Q' <- E [ r +  max_a  [Q(s_t+1, a_t+1)] ]   (sampling to approximate expectation)
        # where Q is our previous q_model

        # by using a function approximator, there is, in theory no guaranteed convergence
        # Empirically however, it works

        q_max = torch.max(q_model(state_next), dim=1).values
        q_max = q_max*(1-done)  # zeroing out when there is no state_next

        target = q_model(state)
        # modifying the target in the q action that was actually performed
        target = target.scatter_(1, action.unsqueeze(1), (reward + self.discount_factor*q_max).unsqueeze(1))

        q_model.fit(state, target)

    def learn_policy(self) -> Policy:
        episodes = self.hyperparameters["episodes_to_train"]
        agent = Agent(self.environment, self.exploration_policy)

        score = RewardObserver()
        agent.attach_observer(self.trajectories)
        agent.attach_observer(score)

        # TODO Fix q_model so as not to accumulate gradient during act, only on experience replay

        for episode in range(episodes):
            agent.perform_episode(after_each_step=[self.experience_replay])

        score.plot()

        return GreedyQPolicy(q_model=self.q_model)
