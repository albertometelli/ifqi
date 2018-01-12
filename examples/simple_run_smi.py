
import time
import pickle
import numpy as np
from gym.utils import seeding
import matplotlib.pyplot as plt

from spmi.utils.uniform_policy import UniformPolicy
from spmi.utils.random_policy import RandomPolicy
from spmi.envs.race_track_configurable import RaceTrackConfigurableEnv
from spmi.algorithms.smi_exact_par import pSMI


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
path_file = "/Simulazione_pSMI"

policy_file = "/Users/mirco/Desktop/Simulazioni/policy_track0wall"

startTime = time.time()

k = 0.5

mdp = RaceTrackConfigurableEnv(track_file='track0wall', initial_configuration=k)

print('nS: {0}'.format(mdp.nS))

print('MDP instantiated')

policy = pickle.load(open(policy_file, 'rw'))

eps = 0.0001

smi = pSMI(mdp, eps)
w = np.array([k, 1 - k])

model = smi.parametric_safe_model_iteration(policy, w)



iterations = np.array(range(smi.iteration))
evaluations = np.array(smi.evaluations)
advantages = np.array(smi.advantages)
distances_sup = np.array(smi.distances_sup)
distances_mean = np.array(smi.distances_mean)
betas = np.array(smi.betas)

plt.switch_backend('pdf')

plt.figure()
plt.title("Performance")
plt.xlabel('Iteration')
plt.plot(iterations, evaluations, label='J_P')
plt.legend(loc='best', fancybox=True)
plt.savefig(path_name + path_file + "/performance")


plt.figure()
plt.title('Distance')
plt.xlabel('Iteration')
plt.plot(iterations, distances_sup, label='distance_sup')
plt.plot(iterations, distances_mean, label='distance_mean')
plt.legend(loc='best', fancybox=True)
plt.savefig(path_name + path_file + "/distance")

plt.figure()
plt.title('Convex combination coefficient')
plt.xlabel('Iteration')
plt.plot(iterations, distances_sup, label='coefficient')
plt.legend(loc='best', fancybox=True)
plt.savefig(path_name + path_file + "/coefficient")

plt.figure()
plt.title('Advantage and beta coefficient')
plt.xlabel('Iteration')
plt.plot(iterations, advantages, label='advantage')
plt.plot(iterations, betas, label='beta')
plt.legend(loc='best', fancybox=True)
plt.savefig(path_name + path_file + "/betas_advantage")


print('The script took {0} minutes'.format((time.time() - startTime) / 60))