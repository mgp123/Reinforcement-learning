import torch
from tqdm import tqdm

from src.agent_observer import ReplayBuffer
from src.epsilon_greedy import GreedyQPolicy
from src.learner import *
from src.policy import Policy


class DQN(Learner):
    def __init__(self, environment, q_model: FunctionApproximatorType, optimizer,
                 exploration_policy: Policy, discount_factor=0.95):
        """
        Performs Q iteration for environment using q_model and exploration_policy.
        q_model is updated

        :param environment :
            This is the MDP that we wish to use reinforcement learning on.
            Should support step.
        :param optimizer to use for q_model
        :param q_model : Function approximation to use. Typically a neural network.
            Should support the () operation. This is though with pytorch in mind.
        :param exploration_policy : Policy to be used by agent to collect data for q iteration.
        """
        super(DQN, self).__init__(environment, discount_factor=discount_factor)
        self.q_model = q_model

        self.hyperparameters = {}

        # if exploration_policy is None:
        #   exploration_policy = DecayingEpsilonGreedyQPolicy(q_model, 1.0, 0.95)

        self.exploration_policy = exploration_policy
        self.optimizer = optimizer
        self.replay_buffer = None

    def experience_replay(self, **kwargs):
        n_samples = self.hyperparameters["experience_replay_samples"]

        if self.replay_buffer.size() < n_samples:
            return

        transitions = self.replay_buffer.sample_transitions(n_samples)
        action, done, reward, state, state_next = self.transitions_to_tensors(transitions)

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

        # TODO Fix q_model so as not to accumulate gradient during act or on experience replay, only on fit
        with torch.no_grad():
            q_max = torch.max(q_model(state_next), dim=1).values
            q_max = q_max * (1 - done)  # zeroing out when there is no state_next

            target = q_model(state)
            # modifying the target in the q action that was actually performed
            target = target.scatter_(1, action.unsqueeze(1), (reward + self.discount_factor * q_max).unsqueeze(1))

        self.fit_model(state, target)

    @staticmethod
    def transitions_to_tensors(transitions):
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

        return action, done, reward, state, state_next

    def learn_policy(self, episodes=200, experience_replay_samples=32, buffer_size=30000) -> Tuple[
        GreedyQPolicy, List[int]]:
        self.hyperparameters["experience_replay_samples"] = experience_replay_samples
        self.replay_buffer = ReplayBuffer(buffer_size)
        environment = self.environment
        exploration_policy = self.exploration_policy

        score = []

        for _ in tqdm(range(episodes)):
            done = False
            state = environment.reset()
            exploration_policy.on_episode_start()
            episode_score = 0

            while not done:
                action = exploration_policy(state)
                state_next, reward, done, info = environment.step(action)

                self.replay_buffer.add_transition(state, action, reward, state_next, done)
                self.experience_replay()

                state = state_next
                episode_score += reward

            exploration_policy.on_episode_end()
            score.append(episode_score)

        return GreedyQPolicy(q_model=self.q_model), score

    def fit_model(self, state, target):
        self.optimizer.zero_grad()

        model_output = self.q_model(state)

        loss = torch.nn.MSELoss()
        loss(model_output, target).backward()

        self.optimizer.step()
