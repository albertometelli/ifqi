
import time
import numpy as np
import matplotlib.pyplot as plt

from spmi.envs.race_track_configurable import RaceTrackConfigurableEnv
from spmi.algorithms.optimal_model_finder import OptimalModelFinder


path_name = "/Users/mirco/Desktop/Simulazioni/ModelPlot"

startTime = time.time()

k = 0.5

mdp = RaceTrackConfigurableEnv(track_file='track0easy', initial_configuration=k, reward_weight=[1., 0., 0., 0., 0.])

print('nS: {0}'.format(mdp.nS))
print('MDP instantiated')

grid_step = 0.05
threshold = 0.0001

omf = OptimalModelFinder(mdp)
omf.optimal_model_finder(grid_step, threshold)


coefficient = np.array(omf.coefficient)
performance = np.array(omf.performance)


#plt.switch_backend('pdf')

#plt.figure()
#plt.title('Optimal Model Performance')
#plt.xlabel('Coefficient')
#plt.ylabel('Performance')
plt.plot(coefficient, performance)
#plt.savefig(path_name + "/track1")


print('The script took {0} minutes'.format((time.time() - startTime) / 60))