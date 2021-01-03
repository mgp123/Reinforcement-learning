from typing import Tuple, List

import torch
from tqdm import tqdm

from agent import Agent
from agent_observer import RewardObserver, TrajectoryObserver
from learner import Learner
from policy import Policy
from policy_gradient import StochasticPolicy
from pytorch_utilities import list_of_tuples_to_tuple_of_tensors


def concatenate(list_of_lists):
    res = []
    for x in list_of_lists:
        res += x
    return res


class ActorCritic(Learner):

    def __init__(self,
                 environment,
                 actor, actor_optimizer,
                 critic, critic_optimizer,
                 discount_factor):
        super(ActorCritic, self).__init__(environment, discount_factor)

        self.actor = actor
        self.actor_optimizer = actor_optimizer

        self.critic = critic
        self.critic_optimizer = critic_optimizer

    def learn_policy(self, *args, **kwargs) -> Tuple[Policy, List[float]]:
        raise NotImplementedError


class PPO(ActorCritic):
    def __init__(self,
                 environment,
                 actor, actor_optimizer,
                 critic=None, critic_optimizer=None,
                 discount_factor=0.95):

        super(PPO, self).__init__(environment,
                                  actor, actor_optimizer,
                                  critic, critic_optimizer,
                                  discount_factor)

        self.use_critic = not (critic is None or critic_optimizer is None)

    def learn_policy(self,
                     epochs,
                     actor_iterations=1,
                     critic_iterations=1,
                     episodes_per_update=1,
                     epsilon_bound=0.2) \
            -> Tuple[Policy, List[float]]:

        policy = StochasticPolicy(self.actor)

        agent = Agent(self.environment, policy)
        r_obs = RewardObserver()
        t_obs = TrajectoryObserver()
        agent.attach_observer(t_obs)
        agent.attach_observer(r_obs)

        for _ in tqdm(range(epochs)):
            # TODO COLLECTING EPISODES CAN BE DONE IN PARALLEL WITH MULTIPLE AGENTS
            for _ in range(episodes_per_update):
                agent.perform_episode()

            reward_to_go = t_obs.reward_to_go(self.discount_factor)
            trajectories = t_obs.sampled_trajectories

            # unify trajectories into single list
            reward_to_go = concatenate(reward_to_go)
            trajectories = concatenate(trajectories)

            # to tensor
            reward_to_go = torch.tensor(reward_to_go)
            trajectories = list_of_tuples_to_tuple_of_tensors(trajectories)

            if self.use_critic:
                state_index = 0

                v = self.critic(trajectories[state_index])
                v = torch.squeeze(v, 1)

                advantage = reward_to_go - v
                advantage = advantage.detach()

                self.update_actor(trajectories, advantage, actor_iterations, epsilon_bound)
                self.update_critic(trajectories, reward_to_go, critic_iterations)
            else:
                self.update_actor(trajectories, reward_to_go, actor_iterations, epsilon_bound)

            # reset memory for next iteration
            t_obs.clear()

        return policy, r_obs.get_rewards()

    def update_actor(self, trajectories, advantage, actor_iterations, epsilon_clip):
        state_index = 0
        action_index = 1

        p_old = self.actor(trajectories[state_index])
        p_old = p_old.log_prob(trajectories[action_index]).exp()
        p_old = p_old.detach()  # set it as constant

        for _ in range(actor_iterations):
            p_new = self.actor(trajectories[state_index])
            p_new = p_new.log_prob(trajectories[action_index]).exp()

            r = p_new / p_old
            loss = torch.min(
                r * advantage,
                torch.clamp(r, min=1 - epsilon_clip, max=1 + epsilon_clip) * advantage
            )

            loss = -loss.mean()
            loss.backward()
            self.actor_optimizer.step()
            self.actor_optimizer.zero_grad()

    def update_critic(self, trajectories, reward_to_go, critic_iterations):
        state = trajectories[0]

        for _ in range(critic_iterations):
            v = self.critic(state)
            v = torch.squeeze(v, 1)

            loss = torch.nn.MSELoss()(v, reward_to_go)
            loss.backward()
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad()
