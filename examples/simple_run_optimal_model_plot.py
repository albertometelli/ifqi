
import time
import numpy as np
import matplotlib.pyplot as plt

from spmi.envs.race_track_configurable_mod import RaceTrackConfigurableEnv
from spmi.algorithms.optimal_model_finder import OptimalModelFinder


path_name = "/Users/mirco/Desktop/Simulazioni/ModelPlot"

startTime = time.time()

plt.switch_backend('pdf')
plt.figure()

k = 0.5
rew = [0.9, -0.1, 0, 0, 0.05]
rew0 = None

max_psucc = 0.8
min_psucc = 0.5
max_psucc2 = 0.62
min_psucc2 = 0.3

grid_step = 0.1
threshold = 0.0001

for i in np.arange(0.6, 0.62 + 0.001, 0.001):

    mdp = RaceTrackConfigurableEnv('track_greek', k, max_psucc, min_psucc, i, min_psucc2, reward_weight=rew)

    omf = OptimalModelFinder(mdp)
    omf.optimal_model_finder(grid_step, threshold)

    coefficient = np.array(omf.coefficient)
    performance = np.array(omf.performance)

    plt.plot(coefficient, performance)


plt.title('Optimal Model Performance')
plt.xlabel('Coefficient')
plt.ylabel('Performance')
plt.savefig(path_name + "/test")


print('The script took {0} minutes'.format((time.time() - startTime) / 60))