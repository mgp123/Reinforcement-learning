class Policy(object):

    def on_episode_start(self):
        pass

    def on_episode_end(self):
        pass

    def __call__(self, state):
        raise NotImplementedError
