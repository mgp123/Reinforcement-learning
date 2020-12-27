import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from agent import Agent
from agent_observer import TrajectoryObserver, RewardObserver
from learner import Learner
from policy import Policy
from policy_gradient import PolicyGradient, StochasticPolicy
from pytorch_utilities import list_of_tuples_to_tuple_of_tensors


class ActorCriticMonteCarloVEstimate(PolicyGradient):
    def __init__(
            self, environment,
            a_distribution_model, a_optimizer,
            v_model, v_optimizer,
            discount_factor=0.95):
        super(ActorCriticMonteCarloVEstimate, self).__init__(
            environment,
            a_distribution_model, a_optimizer,
            discount_factor=discount_factor)

        self.v_model = v_model
        self.v_optimizer = v_optimizer

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

    def learn_policy(self, epochs=200, transition_batch_size=1) -> Policy:
        state_index = 0
        action_index = 1
        self.v_optimizer.zero_grad()

        policy = StochasticPolicy(self.a_distribution_model)
        agent = Agent(self.environment, policy)

        # utilities to collect agent data
        r_obs = RewardObserver()
        agent.attach_observer(r_obs)

        for _ in tqdm(range(epochs)):
            transitions = [agent.yield_transition() for _ in range(transition_batch_size)]
            transitions = list_of_tuples_to_tuple_of_tensors(transitions)

            # update v model by bootstraping
            self.update_v(transitions)

            # calculate loss
            advantage = self.get_advantage(transitions)
            policy_loss = self.a_distribution_model(transitions[state_index])
            policy_loss = - policy_loss.log_prob(transitions[action_index]) * advantage
            policy_loss = torch.sum(policy_loss)

            # gradient step
            policy_loss.backward()
            self.a_optimizer.step()
            self.a_optimizer.zero_grad()

        plt.plot(r_obs.get_rewards())
        plt.xlabel('episode')
        plt.ylabel('total reward')
        plt.savefig("score.png")

        return policy

    def get_advantage(self, transitions):
        # TODO complete algorithm
        raise NotImplementedError

    def update_v(self, transitions_tensor):
        raise NotImplementedError


