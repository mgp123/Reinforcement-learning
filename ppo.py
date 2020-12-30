from typing import Tuple, List

import torch
from tqdm import tqdm

from actor_critic import ActorCritic, ActorCriticMonteCarloVEstimate
from agent import Agent
from agent_observer import RewardObserver
from policy import Policy
from policy_gradient import StochasticPolicy
from pytorch_utilities import list_of_tuples_to_tuple_of_tensors


class PPO(ActorCritic):
    def learn_policy(self,
                     epochs,
                     actor_iterations, critic_iterations,
                     transition_batch,
                     epsilon_bound=0.2, v_initialization_episodes=20) \
            -> Tuple[Policy, List[float]]:

        policy = StochasticPolicy(self.a_distribution_model)
        agent = Agent(self.environment, policy)
        r_obs = RewardObserver()
        agent.attach_observer(r_obs)

        # runs montecarlo policy gradient, tuning v_model and a_model
        previous_rewards = self.initialize_v(v_initialization_episodes)
        r_obs.add(previous_rewards)

        transition_generator = agent.transition_generator()

        for _ in tqdm(range(epochs)):
            # TODO THIS LINE CAN BE DONE IN PARALLEL WITH MULTIPLE AGENTS
            transitions = [next(transition_generator) for _ in range(transition_batch)]

            transitions = list_of_tuples_to_tuple_of_tensors(transitions)

            self.update_critic(transitions, critic_iterations)
            advantage = self.get_advantage(transitions)

            self.update_actor(transitions, advantage, actor_iterations, epsilon_bound)

        transition_generator.close()

        return policy, r_obs.get_rewards()

    def update_critic(self, transitions, critic_iterations):
        state_index = 0
        action_index = 1
        reward_index = 2
        state_next_index = 3
        done_index = 4

        # last state_next needs to be appended
        state = torch.zeros(
            transitions[state_index].size()[0] + 1,
            transitions[state_index].size()[1])
        state[:-1] = transitions[state_index]
        state[-1] = transitions[state_next_index][-1]

        for _ in range(critic_iterations):
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

    def get_advantage(self, transitions):
        state_index = 0
        action_index = 1
        reward_index = 2
        state_next_index = 3
        done_index = 4

        # last state_next needs to be appended
        state = torch.zeros(
            transitions[state_index].size()[0] + 1,
            transitions[state_index].size()[1])
        state[:-1] = transitions[state_index]
        state[-1] = transitions[state_next_index][-1]

        v = self.v_model(state)
        v_next = v[1:]
        v = v[:-1]

        # bootstraping
        boosttrapped_v = v_next
        boosttrapped_v = boosttrapped_v * (1 - transitions[done_index])  # zeroing v_next entry  if done
        boosttrapped_v = transitions[reward_index] + self.discount_factor * boosttrapped_v

        advantage = boosttrapped_v - v
        advantage = advantage.detach()  # set it as constant
        return advantage

    def update_actor(self, transitions, advantage, actor_iterations, epsilon_bound):
        state_index = 0
        action_index = 1

        p_old = self.a_distribution_model(transitions[state_index])
        p_old = p_old.log_prob(transitions[action_index]).exp()
        p_old = p_old.detach()  # set it as constant

        for _ in range(actor_iterations):
            p_new = self.a_distribution_model(transitions[state_index])
            p_new = p_new.log_prob(transitions[action_index]).exp()

            r = torch.clamp(p_new / p_old, min=1 - epsilon_bound, max=1 + epsilon_bound)
            loss = r * advantage
            loss = loss.mean()
            loss.backward()
            self.a_optimizer.step()
            self.a_optimizer.zero_grad()

    def initialize_v(self, v_initialization_episodes):
        init_learner = ActorCriticMonteCarloVEstimate(
            self.environment,
            self.a_distribution_model, self.a_optimizer,
            self.v_model, self.v_optimizer,
            self.discount_factor
        )

        _, reward_history = init_learner.learn_policy(epochs=v_initialization_episodes)
        return reward_history
