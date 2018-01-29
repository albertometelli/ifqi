
import time
import numpy as np

from spmi.envs.race_track_configurable_4 import RaceTrackConfigurableEnv
from spmi.algorithms.optimal_model_finder_4 import OptimalModelFinder4
from spmi.utils.tabular import *



startTime = time.time()


grid_num = 11
threshold = 0.00001

mdp = RaceTrackConfigurableEnv(track_file='race_straight', reward_fail_abs=0, pfail=0.07)
# mdp = RaceTrackConfigurableEnv(track_file='track0_start')

print('nS: {0}'.format(mdp.nS))
print('MDP instantiated')

model_set = [TabularModel(mdp.P_highspeed_noboost, mdp.nS, mdp.nA),
             TabularModel(mdp.P_lowspeed_noboost, mdp.nS, mdp.nA),
             TabularModel(mdp.P_highspeed_boost, mdp.nS, mdp.nA),
             TabularModel(mdp.P_lowspeed_boost, mdp.nS, mdp.nA)]

omf = OptimalModelFinder4(mdp, model_set)
optimal_combination, comb, Js = omf.optimal_model_finder(grid_num, threshold)

a, b, c, d = optimal_combination

print('\n----> Optimal vertex combination: hs_nb:{0} ls_nb:{1} hs_b:{2} ls_b:{3}'.format(a, b, c, d))


print('The script took {0} minutes'.format((time.time() - startTime) / 60))

