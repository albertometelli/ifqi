
import time
import numpy as np
import matplotlib.pyplot as plt

from spmi.envs.race_track_configurable import RaceTrackConfigurableEnv
from spmi.algorithms.optimal_model_finder_2 import OptimalModelFinder


path_name = "/Users/mirco/Desktop/Simulazioni/ModelPlot"

startTime = time.time()

plt.switch_backend('pdf')
plt.figure()

k = 0.5
rew = [0.9, -0.1, 0, 0, 0.05]
grid_step = 0.01
threshold = 0.00001

mdp = RaceTrackConfigurableEnv(track_file='track0', initial_configuration=k)
print('nS: {0}'.format(mdp.nS))
print('MDP instantiated')


omf = OptimalModelFinder(mdp)
omf.optimal_model_finder(grid_step, threshold)

coefficient = np.array(omf.coefficient)
performance = np.array(omf.performance)
plt.title('Optimal Model Performance')
plt.xlabel('Coefficient')
plt.ylabel('Performance')
plt.plot(coefficient, performance)
plt.savefig(path_name + "/test")


print('The script took {0} minutes'.format((time.time() - startTime) / 60))