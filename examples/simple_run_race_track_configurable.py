import numpy as np
from ifqi.envs.race_track_configurable import RaceTrackConfigurableEnv
from ifqi.evaluation import evaluation

class RandomPolicy(object):

    def draw_action(self, state, done):
        return np.random.choice(5)


mdp = RaceTrackConfigurableEnv(track_file='track1', initial_configuration=0.5)
policy = RandomPolicy()
trajectories = evaluation.collect_episodes(mdp, policy,  n_episodes=10)
