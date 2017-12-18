import matplotlib.pyplot as plt
import numpy as np
import time

from ifqi.utils.uniform_policy import UniformPolicy
from ifqi.utils.random_policy import RandomPolicy
from ifqi.envs.race_track_configurable import RaceTrackConfigurableEnv
from ifqi.algorithms.spi_exact import SPI
from ifqi.algorithms.parametric_smi_exact import pSMI

path_name = "/Users/mirco/Desktop/Simulazioni"
path_file = "/Simulazione_pSMI"

startTime = time.time()

k = 0.5

mdp = RaceTrackConfigurableEnv(track_file='track0', initial_configuration=k)

print('nS: {0}'.format(mdp.nS))

print('MDP instantiated')

policy_uniform = UniformPolicy(mdp)
policy_random = RandomPolicy(mdp)

eps = 0.0005

spi = SPI(mdp, eps, 0.9)
policy = spi.safe_policy_iteration_target_trick(policy_uniform)

eps = 0.0001

smi = pSMI(mdp, eps)
w = np.array([k, 1 - k])

model = smi.safe_parametric_model_iteration(policy, w)



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