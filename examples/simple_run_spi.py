import matplotlib.pyplot as plt
import numpy as np
import time

from ifqi.utils.uniform_policy import UniformPolicy
from ifqi.utils.random_policy import RandomPolicy
from ifqi.envs.race_track_configurable import RaceTrackConfigurableEnv
from ifqi.algorithms.spi_approximate import SPIa
from ifqi.algorithms.spi_exact import SPI

path_name = "/Users/mirco/Desktop/Simulazioni"
path_file = "/Simulazione_TargetTrick_Comparison"

startTime = time.time()

mdp = RaceTrackConfigurableEnv(track_file='track0', initial_configuration=0.5)

print('nS: {0}'.format(mdp.nS))

print('MDP instantiated')

policy_uniform = UniformPolicy(mdp)
policy_random = RandomPolicy(mdp)

eps = 0.0005
delta = 0.1
SPI = SPI(mdp, eps, delta)

policy = SPI.safe_policy_iteration_target_trick(policy_uniform)


iterations = np.array(range(SPI.iteration))
evaluations = np.array(SPI.evaluations)
advantages = np.array(SPI.advantages)
distances_sup = np.array(SPI.distances_sup)
distances_mean = np.array(SPI.distances_mean)
alfas = np.array(SPI.alfas)


plt.switch_backend('pdf')


plt.figure()
plt.title("Performance")
plt.xlabel('Iteration')
plt.plot(iterations, evaluations, label='J_pi')
plt.legend(loc='best', fancybox=True)
plt.savefig(path_name + path_file + "/performance")


plt.figure()
plt.title('Distances')
plt.xlabel('Iteration')
plt.plot(iterations, distances_sup, label='distance_sup')
plt.plot(iterations, distances_mean, label='distance_mean')
plt.legend(loc='best', fancybox=True)
plt.savefig(path_name + path_file + "/distances")

plt.figure()
plt.title('Advantage and alfa coefficient')
plt.xlabel('Iteration')
plt.plot(iterations, advantages, label='advantage')
plt.plot(iterations, alfas, label='alfa')
plt.legend(loc='best', fancybox=True)
plt.savefig(path_name + path_file + "/alfa_advantage")


print('The script took {0} minutes'.format((time.time() - startTime) / 60))