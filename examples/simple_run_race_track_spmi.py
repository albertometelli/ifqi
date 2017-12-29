from spmi.envs.race_track_configurable import RaceTrackConfigurableEnv
from spmi.evaluation.evaluation import collect_episodes
from spmi.algorithms.spmi_exact_non_par import SPMI
from spmi.utils.uniform_policy import UniformPolicy
from spmi.utils.tabular import TabularPolicy, TabularModel
from spmi.algorithms.policy_chooser import *
from spmi.algorithms.model_chooser import *
import numpy as np
from gym.utils import seeding
import matplotlib.pyplot as plt

mdp = RaceTrackConfigurableEnv(track_file='track1', initial_configuration=0.5)
uniform_policy = UniformPolicy(mdp)

initial_model = TabularModel(mdp.P, mdp.nS, mdp.nA)
initial_policy = TabularPolicy(uniform_policy.get_rep(), mdp.nS, mdp.nA)

model_set = [TabularModel(mdp.P1, mdp.nS, mdp.nA), TabularModel(mdp.P2, mdp.nS, mdp.nA)]

policy_chooser = GreedyPolicyChooser(mdp.nS, mdp.nA)
model_chooser = SetPolicyChooser(model_set, mdp.nS, mdp.nA)