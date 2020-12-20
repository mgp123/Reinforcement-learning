
class Agent(object):

    def __init__(self, environment, policy):
        """
            Parameters
            ----------
            environment : should support step
                the environment in which the agent interacts. ie acts upon and receives states and rewards
            policy :
                Dictates how the agent must react to each state.
        """

        self.environment = environment
        self.policy = policy

    def act(self, state):
        # should return action to perform
        return self.policy(state)

    def perform_episode(self, before_start_of_episode=[], after_each_step=[], after_end_of_episode=[]):
        state = self.environment.reset()
        done = False

        # before start of episode procedures
        for f in before_start_of_episode:
            f(initial_state=state)
        self.policy.on_episode_start()

        # episode loop
        while not done:
            action = self.act(state)
            state_next, reward, done, info = self.environment.step(action)

            # after each step procedures
            for f in after_each_step:
                f(state=state, action=action, state_next=state_next, reward=reward, done=done, info=info)

            state = state_next

        for f in after_end_of_episode:
            f()

        self.policy.on_episode_end()

    def set_policy(self, policy):
        self.policy = policy
