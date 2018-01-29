
import time
import pickle
import numpy as np
from gym.utils import seeding

from spmi.utils.uniform_policy import UniformPolicy
from spmi.utils.random_policy import RandomPolicy
from spmi.envs.race_track_configurable_2 import RaceTrackConfigurableEnv
from spmi.algorithms.spi_exact import SPI


# define SpiPolicy class at module top level to use pickle!

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

###################################################


path_name = "/Users/mirco/Desktop/Simulazioni"
filename = "/policy_track0wall"

startTime = time.time()

mdp = RaceTrackConfigurableEnv(track_file='track0wall', initial_configuration=1)

print('nS: {0}'.format(mdp.nS))

print('MDP instantiated')

policy_uniform = UniformPolicy(mdp)
policy_random = RandomPolicy(mdp)

eps = 0.0005
delta = 0.1
SPI = SPI(mdp, eps, delta)

# policy = pickle.load(open(path_name + filename, 'rw'))

policy = SPI.safe_policy_iteration_target_trick(policy_uniform)

filehandler = open(path_name + filename, 'w')
pickle.dump(policy, filehandler)


print('The script took {0} minutes'.format((time.time() - startTime) / 60))