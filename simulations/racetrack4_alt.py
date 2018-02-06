from spmi.envs.race_track_configurable_4 import RaceTrackConfigurableEnv
from spmi.utils.uniform_policy import UniformPolicy
from spmi.algorithms.spmi_exact_non_par import SPMI
from spmi.algorithms.policy_chooser import *
from spmi.algorithms.model_chooser import *
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import os

track = 'race_straight'
simulation_name = 'racetrack4_comparison'
dir_path = "/home/deep/mirco/spmi/simulations/data/" + simulation_name

if not os.path.exists(dir_path):
    os.makedirs(dir_path)

mdp = RaceTrackConfigurableEnv(track_file=track, initial_configuration=[0.5, 0.5, 0, 0], pfail=0.07)

original_model = copy.deepcopy(mdp.P)

uniform_policy = UniformPolicy(mdp)

initial_model = TabularModel(mdp.P, mdp.nS, mdp.nA)
initial_policy = TabularPolicy(uniform_policy.get_rep(), mdp.nS, mdp.nA)

model_set = [TabularModel(mdp.P_highspeed_noboost, mdp.nS, mdp.nA),
             TabularModel(mdp.P_lowspeed_noboost, mdp.nS, mdp.nA),
             TabularModel(mdp.P_highspeed_boost, mdp.nS, mdp.nA),
             TabularModel(mdp.P_lowspeed_boost, mdp.nS, mdp.nA)]

policy_chooser = GreedyPolicyChooser(mdp.nS, mdp.nA)
model_chooser = SetModelChooser(model_set, mdp.nS, mdp.nA)

eps = 0.0
spmi = SPMI(mdp, eps, policy_chooser, model_chooser, max_iter=100000, use_target_trick=True, delta_q=1)

#-------------------------------------------------------------------------------
#SPMI alternated

spmi.spmi_alt(initial_policy, initial_model)

alt_iterations = np.array(range(spmi.iteration))
alt_evaluations = np.array(spmi.evaluations)
alt_p_advantages = np.array(spmi.p_advantages)
alt_m_advantages = np.array(spmi.m_advantages)
alt_p_dist_sup = np.array(spmi.p_dist_sup)
alt_p_dist_mean = np.array(spmi.p_dist_mean)
alt_m_dist_sup = np.array(spmi.m_dist_sup)
alt_m_dist_mean = np.array(spmi.m_dist_mean)
alt_alfas = np.array(spmi.alfas)
alt_betas = np.array(spmi.betas)
alt_p_change = np.cumsum(1 - np.array(spmi.p_change))
alt_m_change = np.cumsum(1 - np.array(spmi.m_change))
alt_coefficient = np.array(spmi.w_current)
alt_bound = np.array(spmi.bound)
spmi.save_simulation(dir_path, 'alt.csv')