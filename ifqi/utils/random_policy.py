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

class RandomPolicy(object):
    """
    Class to instantiate a policy which selects
    an action randomly on the action space
    """

    def __init__(self, mdp):
        """
        The constructor returns a policy as a dictionary over states
        :param mdp: the environment on which the method construct the policy
        """
        self._seed()
        self.mdp = mdp
        nS = mdp.nS
        nA = mdp.nA
        policy = {s: [] for s in range(nS)}
        for s in range(nS):
            actions = np.zeros(nA)
            valid_actions = mdp.get_valid_actions(s)
            for a in valid_actions:
                prob = np.random.random()
                actions[a] = prob
            actions = actions / np.sum(actions)
            policy[s] = actions

        self.policy = policy


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def draw_action(self, state, done):
        actions = self.policy[np.asscalar(state)]
        i = categorical_sample(actions, self.np_random)
        return i

    def get_rep(self):
        return self.policy