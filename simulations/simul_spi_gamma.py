import matplotlib.pyplot as plt
import numpy as np
import time

from spmi.utils.uniform_policy import UniformPolicy
from spmi.utils.random_policy import RandomPolicy
from spmi.envs.race_track_configurable import RaceTrackConfigurableEnv
from spmi.algorithms.spi_exact import SPI

path_name = "/Users/mirco/Desktop/Simulazioni"
path_file = "/Simulazione_SPI_Target_Trick_track0"

startTime = time.time()

track = 'track0'
mdp = RaceTrackConfigurableEnv(track_file=track, initial_configuration=0.5)
mdp.set_gamma(0.9)

print('nS: {0}'.format(mdp.nS))

print('MDP instantiated')

policy_uniform = UniformPolicy(mdp)
policy_random = RandomPolicy(mdp)

eps = 0.0002
delta = 0.1
spi = SPI(mdp, eps, delta)
spi_trick = SPI(mdp, eps, delta)


policy_target_trick = spi_trick.safe_policy_iteration_target_trick(policy_uniform)

iterations_t = np.array(range(spi_trick.iteration))
performance_t = np.array(spi_trick.evaluations)
advantage_t = np.array(spi_trick.advantages)
distance_sup_t = np.array(spi_trick.distances_sup)
distance_mean_t = np.array(spi_trick.distances_mean)
alfa_t = np.array(spi_trick.alfas)


plt.switch_backend('pdf')


plt.figure()
plt.title("Performance: eSPI target trick")
plt.xlabel('Iteration')
plt.plot(iterations_t, performance_t, color='tab:blue', linestyle='dashed', label='J eSPItt')
plt.legend(loc='best', fancybox=True)
plt.savefig(path_name + path_file + "/performance")


plt.figure()
plt.title('Sup distance and mean distance: eSPI target trick')
plt.xlabel('Iteration')
plt.plot(iterations_t, distance_sup_t, color='tab:blue', linestyle='dashed', label='sup eSPItt ')
plt.plot(iterations_t, distance_mean_t, color='tab:orange', linestyle='dashed', label='mean eSPItt')
plt.legend(loc='best', fancybox=True)
plt.savefig(path_name + path_file + "/distances")


plt.figure()
plt.title('Advantage and alfa coefficient: eSPI target trick')
plt.xlabel('Iteration')
plt.plot(iterations_t, advantage_t, color='tab:blue', linestyle='dashed', label='adv eSPItt')
plt.plot(iterations_t, alfa_t, color='tab:orange', linestyle='dashed', label='alfa eSPItt')
plt.legend(loc='best', fancybox=True)
plt.savefig(path_name + path_file + "/alfa_advantage")


print('The script took {0} minutes'.format((time.time() - startTime) / 60))