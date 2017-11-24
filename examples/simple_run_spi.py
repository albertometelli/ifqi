import numpy as np
import time
from ifqi.utils.uniform_policy import UniformPolicy
from ifqi.utils.random_policy import RandomPolicy
from ifqi.envs.race_track_configurable import RaceTrackConfigurableEnv
from ifqi.algorithms.spi import SafePolicyIterator


startTime = time.time()

mdp = RaceTrackConfigurableEnv(track_file='track1', initial_configuration=0.5)

print('MDP instantiated')

policy_uniform = UniformPolicy(mdp)
policy_random = RandomPolicy(mdp)
SPI = SafePolicyIterator(mdp, 0.1, 0.1)
# er_advantage, greedy_policy = SPI.pol_chooser(policy_uniform)
# distance = SPI.pol_distance(greedy_policy, policy_uniform)

policy = SPI.safe_policy_iteration(policy_uniform)

print('The script took {0} seconds'.format(time.time() - startTime))