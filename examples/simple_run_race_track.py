import numpy as np
from ifqi.envs.race_track import RaceTrackEnv
from ifqi.evaluation import evaluation

class RandomPolicy(object):

    def draw_action(self, state, done):
        return np.random.choice(5)


mdp = RaceTrackEnv(track_file='track1')
policy = RandomPolicy()
trajectories = evaluation.collect_episodes(mdp, policy,  n_episodes=10)
