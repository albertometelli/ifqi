import numpy as np
import time
from spmi.envs.race_track_configurable import RaceTrackConfigurableEnv
from spmi.evaluation import evaluation

class RandomPolicy(object):

    def draw_action(self, state, done):
        return np.random.choice(5)

startTime = time.time()

mdp = RaceTrackConfigurableEnv(track_file='track1', initial_configuration=0.5)
policy = RandomPolicy()
trajectories = evaluation.collect_episodes(mdp, policy,  n_episodes=10)

print('The script took {0} seconds'.format(time.time() - startTime))