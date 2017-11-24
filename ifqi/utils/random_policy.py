import numpy as np

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

        self.mdp = mdp
        self.nA = mdp.nA

    def draw_action(self, state, done):
        return np.random.choice(self.nA)