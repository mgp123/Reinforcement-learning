from src.policy import Policy
from src.type_definitions import StateType


class Agent(object):

    def __init__(self, environment, policy: Policy):
        """
        Agent carries out interactions with the environment according to its current policy

        :param environment:
            The environment in which the agent interacts. ie acts upon and receives states and rewards.
            Should support step.
        :param policy: Dictates how the agent must react to each state.
        """

        self.current_state = None
        self.environment = environment
        self.policy = policy
        self.observers = []

    def act(self, state: StateType):
        return self.policy(state)

    def perform_episode(self, render=False):
        self.start_episode()
        done = False

        while not done:
            if render:
                self.environment.render()

            _, _, _, _, done = self.step()

        self.end_episode()  # this is unnecessary as the step already does this but for completeness...

    def end_episode(self):
        # do not end episode if there is none
        if self.current_state is not None:
            self.current_state = None
            self.policy.on_episode_end()
            self.inform_observers_episode_end()

    def start_episode(self):
        self.end_episode()  # end episode if currently in one

        self.current_state = self.environment.reset()
        self.policy.on_episode_start()
        self.inform_observers_episode_start()

    def step(self):
        # if not currently in an episode, start it
        if self.current_state is None:
            self.start_episode()

        state = self.current_state
        action = self.act(state)
        state_next, reward, done, info = self.environment.step(action)

        self.current_state = state_next

        self.inform_observers_step(state, action, state_next, reward, done, info)

        # if episode done finalize it
        if done:
            self.end_episode()

        return state, action, reward, state_next, done

    def transition_generator(self):
        """
        Infinite transition generator
        :return: (state, action, reward, state_next, done)
        """
        while True:
            done = False

            self.start_episode()

            while not done:
                state, action, reward, state_next, done = self.step()
                yield state, action, reward, state_next, done

            self.end_episode()

    def set_policy(self, policy: Policy):
        self.policy = policy

    def attach_observer(self, observer):
        self.observers.append(observer)

    def inform_observers_episode_start(self):
        for o in self.observers:
            o.on_episode_start(initial_state=self.current_state)

    def inform_observers_episode_end(self):
        for o in self.observers:
            o.on_episode_end()

    def inform_observers_step(self, state, action, state_next, reward, done, info):
        for o in self.observers:
            o.on_step(state=state, action=action, state_next=state_next, reward=reward, done=done, info=info)
