import torch

from agent_observer import TrajectoryObserver
from pytorch_utilities import get_reward_to_go, list_of_tuples_to_tuple_of_tensors

# TODO IMPRACTICAL AS IF USED WOULD PROBABLY CAUSE OTHER PARTS RECALCULATE V UNNECESSARY


class MonteCarloVEstimator(TrajectoryObserver):
    def __init__(self, v_model, v_optimizer, discount_factor, automatic_update=True, episodes_to_update=1):
        super().__init__()
        self.automatic_update = automatic_update
        self.v_model = v_model
        self.v_optimizer = v_optimizer
        self.episodes_to_update = episodes_to_update
        self.discount_factor = discount_factor

    def on_episode_end(self):
        super(MonteCarloVEstimator, self).on_episode_end()

        past_episodes = len(self.sampled_trajectories)
        trajectory = self.last_trajectory()
        reward_to_go = get_reward_to_go(trajectory, self.discount_factor)

        # convert to pytorch tensors
        trajectory = list_of_tuples_to_tuple_of_tensors(trajectory)
        reward_to_go = torch.tensor(reward_to_go, dtype=torch.float32)

        state_index = 0
        v = self.v_model(trajectory[state_index])

        # accumulate gradient to update v model later
        # using reward to go, as estimate for V value
        loss = torch.nn.MSELoss()(v, reward_to_go.unsqueeze(1))
        loss.backward()

        if past_episodes == self.episodes_to_update and self.automatic_update:
            self.fit()

    def fit(self):
        self.v_optimizer.step()
        self.v_optimizer.zero_grad()
        self.clear()
