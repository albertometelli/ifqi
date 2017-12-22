import numpy as np
from gym.utils import seeding


def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


class SpiPolicy(object):
    """
    Class to instantiate a policy with given policy representation
    """

    def __init__(self, rep):
        """
        The constructor instantiate a policy as a dictionary over states
        in which each state point to an array of probability over actions
        :param rep: policy representation as a dictionary over states
        """
        self.policy = rep
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def draw_action(self, state, done):
        actions = self.policy[np.asscalar(state)]
        i = categorical_sample(actions, self.np_random)
        return i

    def get_rep(self):
        return self.policy