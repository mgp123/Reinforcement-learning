from policy import Policy
from type_definitions import StateType


class Agent(object):

    def __init__(self, environment, policy: Policy):
        """
        Agent carries out interactions with the environment according to its current policy

        :param environment:
            The environment in which the agent interacts. ie acts upon and receives states and rewards.
            Should support step.
        :param policy: Dictates how the agent must react to each state.
        """

        self.environment = environment
        self.policy = policy
        self.observers = []

    def act(self, state: StateType):
        # should return action to perform
        return self.policy(state)

    def perform_episode(self, render=False, before_start_of_episode=[], after_each_step=[], after_end_of_episode=[]):
        state = self.environment.reset()
        done = False

        # before start of episode procedures
        #  --------------------------------
        self.start_episode(before_start_of_episode, state)

        # episode loop
        #  --------------------------------
        while not done:
            if render:
                self.environment.render()

            state, action, reward, state_next, done = self.perform_step(state, after_each_step)

            state = state_next

        # after end of episode procedures
        #  --------------------------------
        self.end_episode(after_end_of_episode)

    def end_episode(self, after_end_of_episode):
        self.policy.on_episode_end()
        self.inform_observers_episode_end()
        for f in after_end_of_episode:
            f()

    def start_episode(self, before_start_of_episode, state):
        self.policy.on_episode_start()
        self.inform_observers_episode_start(initial_state=state)
        for f in before_start_of_episode:
            f(initial_state=state)

    def perform_step(self, state, after_each_step=[]):
        action = self.act(state)
        state_next, reward, done, info = self.environment.step(action)
        # after each step procedures
        #  --------------------------------
        self.inform_observers_step(state, action, state_next, reward, done, info)
        for f in after_each_step:
            f(state=state, action=action, state_next=state_next, reward=reward, done=done, info=info)

        return state, action, reward, state_next, done

    def yield_transition(self, before_start_of_episode=[], after_each_step=[], after_end_of_episode=[]):
        """
        Infinite transition generator
        :return: (state, action, reward, state_next, done)
        """
        while True:
            done = False
            state = self.environment.reset()

            self.start_episode(before_start_of_episode, state)

            while not done:
                state, action, reward, state_next, done = self.perform_step(state, after_each_step)
                yield state, action, reward, state_next, done

                state = state_next

            self.end_episode(after_end_of_episode)

    def set_policy(self, policy: Policy):
        self.policy = policy

    def attach_observer(self, observer):
        self.observers.append(observer)

    def inform_observers_episode_start(self, initial_state):
        for o in self.observers:
            o.on_episode_start(initial_state=initial_state)

    def inform_observers_episode_end(self):
        for o in self.observers:
            o.on_episode_end()

    def inform_observers_step(self, state, action, state_next, reward, done, info):
        for o in self.observers:
            o.on_step(state=state, action=action, state_next=state_next, reward=reward, done=done, info=info)
