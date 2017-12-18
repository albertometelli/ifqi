import matplotlib.pyplot as plt
import numpy as np
import time

from ifqi.utils.uniform_policy import UniformPolicy
from ifqi.utils.random_policy import RandomPolicy
from ifqi.envs.race_track_configurable import RaceTrackConfigurableEnv
from ifqi.algorithms.spi_exact import SPI

path_name = "/Users/mirco/Desktop/Simulazioni"
path_file = "/Simulazione_Sup_Mean_Comparison"

startTime = time.time()

track = 'track0'
mdp = RaceTrackConfigurableEnv(track_file=track, initial_configuration=0.5)

print('nS: {0}'.format(mdp.nS))

print('MDP instantiated')

policy_uniform = UniformPolicy(mdp)
policy_random = RandomPolicy(mdp)

eps = 0.0005
delta = 0.1
spi_sup_sup = SPI(mdp, eps, delta)
spi_sup_mean = SPI(mdp, eps, delta)


policy_sup_mean = spi_sup_mean.safe_policy_iteration_target_trick(policy_uniform)

iterations_t = np.array(range(spi_sup_mean.iteration))
performance_t = np.array(spi_sup_mean.evaluations)
advantage_t = np.array(spi_sup_mean.advantages)
distance_sup_t = np.array(spi_sup_mean.distances_sup)
distance_mean_t = np.array(spi_sup_mean.distances_mean)
alfa_t = np.array(spi_sup_mean.alfas)


policy_sup_sup = spi_sup_sup.safe_policy_iteration_target_trick_sup(policy_uniform)

iterations_nt = np.array(range(spi_sup_sup.iteration))
performance_nt = np.array(spi_sup_sup.evaluations)
advantage_nt = np.array(spi_sup_sup.advantages)
distance_sup_nt = np.array(spi_sup_sup.distances_sup)
distance_mean_nt = np.array(spi_sup_sup.distances_mean)
alfa_nt = np.array(spi_sup_sup.alfas)


plt.switch_backend('pdf')


plt.figure()
plt.title("Performance comparison between eSPItt sup*mean and sup^2")
plt.xlabel('Iteration')
plt.plot(iterations_nt, performance_nt, color='tab:blue', label='J eSPI sup^2')
plt.plot(iterations_t, performance_t, color='tab:blue', linestyle='dashed', label='J eSPI sup*mean')
plt.legend(loc='best', fancybox=True)
plt.savefig(path_name + path_file + "/performance")


plt.figure()
plt.title('Sup distance and mean distance comparison\n between eSPItt sup*mean and sup^2')
plt.xlabel('Iteration')
plt.plot(iterations_nt, distance_sup_nt, color='tab:blue', label='eSPI sup^2')
plt.plot(iterations_t, distance_sup_t, color='tab:blue', linestyle='dashed', label='sup eSPI sup*mean ')
plt.plot(iterations_t, distance_mean_t, color='tab:orange', linestyle='dashed', label='mean eSPI sup*mean')
plt.legend(loc='best', fancybox=True)
plt.savefig(path_name + path_file + "/distances")


plt.figure()
plt.title('Advantage and alfa coefficient comparison\n between eSPItt sup*mean and sup^2')
plt.xlabel('Iteration')
plt.plot(iterations_nt, advantage_nt, color='tab:blue', label='adv eSPI sup^2')
plt.plot(iterations_nt, alfa_nt, color='tab:orange', label='alfa eSPI sup^2')
plt.plot(iterations_t, advantage_t, color='tab:blue', linestyle='dashed', label='adv eSPI sup*mean')
plt.plot(iterations_t, alfa_t, color='tab:orange', linestyle='dashed', label='alfa eSPI sup*mean')
plt.legend(loc='best', fancybox=True)
plt.savefig(path_name + path_file + "/alfa_advantage")


print('The script took {0} minutes'.format((time.time() - startTime) / 60))