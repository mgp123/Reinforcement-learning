import torch
from tqdm import tqdm

from src.agent import Agent
from src.agent_observer import RewardObserver, TrajectoryObserver
from src.learner import Learner
from src.stochastic_policy import StochasticPolicy
from src.pytorch_utilities import list_of_tuples_to_tuple_of_tensors, get_reward_to_go


class ActorCritic(Learner):
    def __init__(
            self, environment,
            a_distribution_model, a_optimizer,
            v_model, v_optimizer,
            discount_factor=0.95):
        """
        Abstract class for policy gradient methods using a V value net and an Action distribution net
        """
        super(ActorCritic, self).__init__(environment, discount_factor)

        self.a_distribution_model = a_distribution_model
        self.a_optimizer = a_optimizer

        self.v_model = v_model
        self.v_optimizer = v_optimizer

    def learn_policy(self, *args, **kwargs):
        raise NotImplementedError


class ActorCriticMonteCarloVEstimate(ActorCritic):

    def update_advantage(self):
        self.v_optimizer.step()
        self.v_optimizer.zero_grad()

    def get_advantage(self, trajectory_torch, reward_to_go_tensor):
        state_index = 0
        reward_index = 2

        v = self.v_model(trajectory_torch[state_index])

        # accumulate gradient to update v model later
        # using reward to go, as estimate for V value
        loss = torch.nn.MSELoss()(v, reward_to_go_tensor.unsqueeze(1))
        loss.backward()

        v_next = torch.zeros(v.shape)
        # last v_next remains zero as there is no next state
        v_next[:-1] = v[1:]

        reward = trajectory_torch[reward_index]

        # advantage = (r_t + discount * V(s_ t+1)) - V(s_t)
        advantage = reward + self.discount_factor * v_next - v

        # to prevent policy gradient to propagate to v_model
        advantage = advantage.detach()

        return advantage

    def learn_policy(self, epochs=200, episodes_per_update=1):
        self.v_optimizer.zero_grad()
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

                advantage = self.get_advantage(trajectory, reward_to_go)

                # calculate loss
                policy_loss = self.a_distribution_model(trajectory[state_index])
                policy_loss = - policy_loss.log_prob(trajectory[action_index]) * advantage
                policy_loss = torch.sum(policy_loss)

                # to estimate the expected gradient of episodes_per_update episodes,
                # we divide the loss by episodes_per_update
                policy_loss = policy_loss / episodes_per_update

                # accumulate gradient
                policy_loss.backward()

            # gradient step
            self.a_optimizer.step()
            self.a_optimizer.zero_grad()
            self.update_advantage()

            t_obs.clear()

        return policy, r_obs.get_rewards()


class ActorCriticBootstrappedVEstimate(Learner):
    def __init__(
            self, environment,
            a_distribution_model, a_optimizer,
            v_model, v_optimizer,
            discount_factor=0.95):
        super(ActorCriticBootstrappedVEstimate, self).__init__(environment, discount_factor=discount_factor)

        self.a_distribution_model = a_distribution_model
        self.a_optimizer = a_optimizer
        self.v_model = v_model
        self.v_optimizer = v_optimizer

    def learn_policy(self, epochs=200, transition_batch_size=2, v_initialization_episodes=20):
        state_index = 0
        action_index = 1
        reward_index = 2
        state_next_index = 3
        done_index = 4

        policy = StochasticPolicy(self.a_distribution_model)
        agent = Agent(self.environment, policy)

        # utilities to collect agent data
        r_obs = RewardObserver()
        agent.attach_observer(r_obs)
        transition_generator = agent.transition_generator()

        # runs montecarlo policy gradient, tuning v_model and a_model
        previous_rewards = self.initialize_v(v_initialization_episodes)
        r_obs.add(previous_rewards)
        print("heyey")

        while len(r_obs.get_rewards()) < epochs:

            # collect transition
            transitions = [next(transition_generator) for _ in range(transition_batch_size)]
            transitions = list_of_tuples_to_tuple_of_tensors(transitions)

            # last state_next needs to be appended
            state = torch.zeros(
                transitions[state_index].size()[0] + 1,
                transitions[state_index].size()[1])
            state[:-1] = transitions[state_index]
            state[-1] = transitions[state_next_index][-1]

            # separate v in v and v_next
            v = self.v_model(state)
            v_next = v[1:]
            v = v[:-1]

            # bootstraping
            boosttrapped_v = v_next
            boosttrapped_v = boosttrapped_v * (1 - transitions[done_index])  # zeroing v_next entry  if done
            boosttrapped_v = transitions[reward_index] + self.discount_factor * boosttrapped_v

            boosttrapped_v = boosttrapped_v.detach()  # set it as constant

            # perform v_model update
            self.v_optimizer.zero_grad()
            v_loss = torch.nn.MSELoss()(v, boosttrapped_v)
            v_loss.backward()
            self.v_optimizer.step()

            # advantage = (r_t + discount * V(s_ t+1)) - V(s_t)
            advantage = boosttrapped_v - v

            # to prevent policy gradient to propagate to v_model
            advantage = advantage.detach()

            policy_loss = self.a_distribution_model(transitions[state_index])
            policy_loss = - policy_loss.log_prob(transitions[action_index]) * advantage
            policy_loss = torch.sum(policy_loss)

            # gradient step
            self.a_optimizer.zero_grad()
            policy_loss.backward()
            self.a_optimizer.step()

        transition_generator.close()

        return policy, r_obs.get_rewards()

    def initialize_v(self, v_initialization_episodes):
        init_learner = ActorCriticMonteCarloVEstimate(
            self.environment,
            self.a_distribution_model, self.a_optimizer,
            self.v_model, self.v_optimizer,
            self.discount_factor
        )

        _, reward_history = init_learner.learn_policy(epochs=v_initialization_episodes)
        return reward_history
