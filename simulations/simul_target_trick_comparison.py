import matplotlib.pyplot as plt
import numpy as np
import time

from spmi.utils.uniform_policy import UniformPolicy
from spmi.utils.random_policy import RandomPolicy
from spmi.envs.race_track_configurable import RaceTrackConfigurableEnv
from spmi.algorithms.spi_exact import SPI

path_name = "/Users/mirco/Desktop/Simulazioni"
path_file = "/Simulazione_TargetTrick_Comparison2"

startTime = time.time()

track = 'track0'
mdp = RaceTrackConfigurableEnv(track_file=track, initial_configuration=0.5)

print('nS: {0}'.format(mdp.nS))

print('MDP instantiated')

policy_uniform = UniformPolicy(mdp)
policy_random = RandomPolicy(mdp)

eps = 0.0001
delta = 0.1
spi = SPI(mdp, eps, delta)
spi_trick = SPI(mdp, eps, delta)


policy_no_trick = spi.safe_policy_iteration(policy_uniform)

iterations_nt = np.array(range(spi.iteration))
performance_nt = np.array(spi.evaluations)
advantage_nt = np.array(spi.advantages)
distance_sup_nt = np.array(spi.distances_sup)
distance_mean_nt = np.array(spi.distances_mean)
alfa_nt = np.array(spi.alfas)


policy_target_trick = spi_trick.safe_policy_iteration_target_trick(policy_uniform)

iterations_t = np.array(range(spi_trick.iteration))
performance_t = np.array(spi_trick.evaluations)
advantage_t = np.array(spi_trick.advantages)
distance_sup_t = np.array(spi_trick.distances_sup)
distance_mean_t = np.array(spi_trick.distances_mean)
alfa_t = np.array(spi_trick.alfas)


plt.switch_backend('pdf')


plt.figure()
plt.title("Performance comparison between eSPI and eSPI + target trick")
plt.xlabel('Iteration')
plt.plot(iterations_nt, performance_nt, color='tab:blue', label='J eSPI')
plt.plot(iterations_t, performance_t, color='tab:blue', linestyle='dashed', label='J eSPI+trick')
plt.legend(loc='best', fancybox=True)
plt.savefig(path_name + path_file + "/performance")


plt.figure()
plt.title('Sup distance and mean distance comparison\n between eSPI and eSPI + target trick')
plt.xlabel('Iteration')
plt.plot(iterations_nt, distance_sup_nt, color='tab:blue', label='sup eSPI')
plt.plot(iterations_nt, distance_mean_nt, color='tab:orange', label='mean eSPI')
plt.plot(iterations_t, distance_sup_t, color='tab:blue', linestyle='dashed', label='sup eSPI+trick ')
plt.plot(iterations_t, distance_mean_t, color='tab:orange', linestyle='dashed', label='mean eSPI+trick')
plt.legend(loc='best', fancybox=True)
plt.savefig(path_name + path_file + "/distances")


plt.figure()
plt.title('Advantage and alfa coefficient comparison\n between eSPI and eSPI + target trick')
plt.xlabel('Iteration')
plt.plot(iterations_nt, advantage_nt, color='tab:blue', label='adv eSPI')
plt.plot(iterations_nt, alfa_nt, color='tab:orange', label='alfa eSPI')
plt.plot(iterations_t, advantage_t, color='tab:blue', linestyle='dashed', label='adv eSPI+trick')
plt.plot(iterations_t, alfa_t, color='tab:orange', linestyle='dashed', label='alfa eSPI+trick')
plt.legend(loc='best', fancybox=True)
plt.savefig(path_name + path_file + "/alfa_advantage")


print('The script took {0} minutes'.format((time.time() - startTime) / 60))