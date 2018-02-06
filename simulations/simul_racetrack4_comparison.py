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
simulation_name = 'simul_4_comparison'
dir_path = "./data/" + simulation_name

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
'''
#-------------------------------------------------------------------------------
#SPMI
spmi.use_target_trick = False
spmi.spmi(initial_policy, initial_model)

iterations = np.array(range(spmi.iteration))
evaluations = np.array(spmi.evaluations)
p_advantages = np.array(spmi.p_advantages)
m_advantages = np.array(spmi.m_advantages)
p_dist_sup = np.array(spmi.p_dist_sup)
p_dist_mean = np.array(spmi.p_dist_mean)
m_dist_sup = np.array(spmi.m_dist_sup)
m_dist_mean = np.array(spmi.m_dist_mean)
alfas = np.array(spmi.alfas)
betas = np.array(spmi.betas)
p_change = np.cumsum(1 - np.array(spmi.p_change))
m_change = np.cumsum(1 - np.array(spmi.m_change))
coefficient = np.array(spmi.w_current)
bound = np.array(spmi.bound)
spmi.save_simulation(dir_path, 'notarget.csv')

#-------------------------------------------------------------------------------
#SPMI sup
mdp.set_initial_configuration(original_model)
spmi.use_target_trick = True
spmi.spmi_sup(initial_policy, initial_model)

sup_iterations = np.array(range(spmi.iteration))
sup_evaluations = np.array(spmi.evaluations)
sup_p_advantages = np.array(spmi.p_advantages)
sup_m_advantages = np.array(spmi.m_advantages)
sup_p_dist_sup = np.array(spmi.p_dist_sup)
sup_p_dist_mean = np.array(spmi.p_dist_mean)
sup_m_dist_sup = np.array(spmi.m_dist_sup)
sup_m_dist_mean = np.array(spmi.m_dist_mean)
sup_alfas = np.array(spmi.alfas)
sup_betas = np.array(spmi.betas)
sup_p_change = np.cumsum(1 - np.array(spmi.p_change))
sup_m_change = np.cumsum(1 - np.array(spmi.m_change))
sup_coefficient = np.array(spmi.w_current)
sup_bound = np.array(spmi.bound)
spmi.save_simulation(dir_path, 'sup.csv')

#-------------------------------------------------------------------------------
#SPMI no full step
mdp.set_initial_configuration(original_model)
spmi.spmi_no_full(initial_policy, initial_model)

int_iterations = np.array(range(spmi.iteration))
int_evaluations = np.array(spmi.evaluations)
int_p_advantages = np.array(spmi.p_advantages)
int_m_advantages = np.array(spmi.m_advantages)
int_p_dist_sup = np.array(spmi.p_dist_sup)
int_p_dist_mean = np.array(spmi.p_dist_mean)
int_m_dist_sup = np.array(spmi.m_dist_sup)
int_m_dist_mean = np.array(spmi.m_dist_mean)
int_alfas = np.array(spmi.alfas)
int_betas = np.array(spmi.betas)
int_p_change = np.cumsum(1 - np.array(spmi.p_change))
int_m_change = np.cumsum(1 - np.array(spmi.m_change))
int_coefficient = np.array(spmi.w_current)
int_bound = np.array(spmi.bound)
spmi.save_simulation(dir_path,  'nofull.csv')
'''
#-------------------------------------------------------------------------------
#SPMI alternated
mdp.set_initial_configuration(original_model)
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
'''
#-------------------------------------------------------------------------------
#SPMI sequential
mdp.set_initial_configuration(original_model)
spmi.spmi_seq_pm(initial_policy, initial_model)

seq_iterations = np.array(range(spmi.iteration))
seq_evaluations = np.array(spmi.evaluations)
seq_p_advantages = spmi.p_advantages
seq_m_advantages = spmi.m_advantages
seq_p_dist_sup = spmi.p_dist_sup
seq_p_dist_mean = spmi.p_dist_mean
seq_m_dist_sup = spmi.m_dist_sup
seq_m_dist_mean = spmi.m_dist_mean
seq_alfas = spmi.alfas
seq_betas = spmi.betas
seq_p_change = np.cumsum(1 - np.array(spmi.p_change))
seq_m_change = np.cumsum(1 - np.array(spmi.m_change))
seq_coefficient = np.array(spmi.w_current)
seq_bound = np.array(spmi.bound)
spmi.save_simulation(dir_path, 'seq_pm.csv')

#-------------------------------------------------------------------------------
#SPMI sequential
mdp.set_initial_configuration(original_model)
spmi.spmi_seq_mp(initial_policy, initial_model)
spmi.save_simulation(dir_path, 'seq_mp.csv')

#-------------------------------------------------------------------------------
#plots

plt.switch_backend('pdf')

plt.figure()
plt.title('Performance')
plt.xlabel('Iteration')
plt.plot(iterations, evaluations, color='b', label='SPMI notarget')
plt.plot(sup_iterations, sup_evaluations, color='y', label='SPMI sup')
plt.plot(alt_iterations, alt_evaluations, color='g', label='alternated')
plt.plot(seq_iterations, seq_evaluations, color='r', label='sequential')
plt.plot(int_iterations, int_evaluations, color='k', linestyle='dotted', label='no full step')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/performance")

plt.figure()
plt.title('P_hs_nb coefficient')
plt.xlabel('Iteration')
plt.plot(iterations, coefficient[:, 0], color='b', label='SPMI notarget')
plt.plot(sup_iterations, sup_coefficient[:, 0], color='y', label='SPMI sup')
plt.plot(alt_iterations, alt_coefficient[:, 0], color='g', label='alternated')
plt.plot(seq_iterations, seq_coefficient[:, 0], color='r', label='sequential')
plt.plot(int_iterations, int_coefficient[:, 0], color='k', linestyle='dotted', label='no full step')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/P_hs_nb")

plt.figure()
plt.title('P_ls_nb coefficient')
plt.xlabel('Iteration')
plt.plot(iterations, coefficient[:, 1], color='b', label='SPMI notarget')
plt.plot(sup_iterations, sup_coefficient[:, 1], color='y', label='SPMI sup')
plt.plot(alt_iterations, alt_coefficient[:, 1], color='g', label='alternated')
plt.plot(seq_iterations, seq_coefficient[:, 1], color='r', label='sequential')
plt.plot(int_iterations, int_coefficient[:, 1], color='k', linestyle='dotted', label='no full step')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/P_ls_nb")

plt.figure()
plt.title('P_hs_b coefficient')
plt.xlabel('Iteration')
plt.plot(iterations, coefficient[:, 2], color='b', label='SPMI notarget')
plt.plot(sup_iterations, sup_coefficient[:, 2], color='y', label='SPMI sup')
plt.plot(alt_iterations, alt_coefficient[:, 2], color='g', label='alternated')
plt.plot(seq_iterations, seq_coefficient[:, 2], color='r', label='sequential')
plt.plot(int_iterations, int_coefficient[:, 2], color='k', linestyle='dotted', label='no full step')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/P_hs_b")

plt.figure()
plt.title('P_ls_b coefficient')
plt.xlabel('Iteration')
plt.plot(iterations, coefficient[:, 3], color='b', label='SPMI notarget')
plt.plot(sup_iterations, sup_coefficient[:, 3], color='y', label='SPMI sup')
plt.plot(alt_iterations, alt_coefficient[:, 3], color='g', label='alternated')
plt.plot(seq_iterations, seq_coefficient[:, 3], color='r', label='sequential')
plt.plot(int_iterations, int_coefficient[:, 3], color='k', linestyle='dotted', label='no full step')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/P_ls_b")

plt.figure()
plt.title('Bound comparison')
plt.xlabel('Iteration')
plt.plot(iterations, bound, color='b', label='SPMI notarget')
plt.plot(sup_iterations, sup_bound, color='y', label='SPMI sup')
plt.plot(alt_iterations, alt_bound, color='g', label='alternated')
plt.plot(seq_iterations, seq_bound, color='r', label='sequential')
plt.plot(int_iterations, int_bound, color='k', linestyle='dotted', label='no full step')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/bound comparison")
'''