from spmi.envs.race_track_configurable_2 import RaceTrackConfigurableEnv
from spmi.utils.uniform_policy import UniformPolicy
from spmi.algorithms.spmi_exact_non_par import SPMI
from spmi.algorithms.policy_chooser import *
from spmi.algorithms.model_chooser import *
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import os


track = 'track0'
simulation_name = 'simul_racetrack2_combination'
dir_path = "/home/deep/mirco/spmi/simulations/data/" + simulation_name

if not os.path.exists(dir_path):
    os.makedirs(dir_path)

mdp = RaceTrackConfigurableEnv(track_file=track, initial_configuration=0.5)


uniform_policy = UniformPolicy(mdp)
original_model = copy.deepcopy(mdp.P)
initial_model = TabularModel(mdp.P, mdp.nS, mdp.nA)
initial_policy = TabularPolicy(uniform_policy.get_rep(), mdp.nS, mdp.nA)

model_set = [TabularModel(mdp.P1, mdp.nS, mdp.nA), TabularModel(mdp.P2, mdp.nS, mdp.nA)]

policy_chooser = GreedyPolicyChooser(mdp.nS, mdp.nA)
model_chooser = SetModelChooser(model_set, mdp.nS, mdp.nA)

eps = 0.0
spmi = SPMI(mdp, eps, policy_chooser, model_chooser, max_iter=10, use_target_trick=True, delta_q=1)

#-------------------------------------------------------------------------------
#SPMI
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
spmi.save_simulation(dir_path, 'spmi.csv')

#-------------------------------------------------------------------------------
#SPMI
mdp.set_initial_configuration(original_model)
spmi.use_target_trick = False
spmi.spmi(initial_policy, initial_model)

ntt_iterations = np.array(range(spmi.iteration))
ntt_evaluations = np.array(spmi.evaluations)
ntt_p_advantages = np.array(spmi.p_advantages)
ntt_m_advantages = np.array(spmi.m_advantages)
ntt_p_dist_sup = np.array(spmi.p_dist_sup)
ntt_p_dist_mean = np.array(spmi.p_dist_mean)
ntt_m_dist_sup = np.array(spmi.m_dist_sup)
ntt_m_dist_mean = np.array(spmi.m_dist_mean)
ntt_alfas = np.array(spmi.alfas)
ntt_betas = np.array(spmi.betas)
ntt_p_change = np.cumsum(1 - np.array(spmi.p_change))
ntt_m_change = np.cumsum(1 - np.array(spmi.m_change))
ntt_coefficient = np.array(spmi.w_current)
ntt_bound = np.array(spmi.bound)
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
spmi.save_simulation(dir_path, 'seq.csv')

#-------------------------------------------------------------------------------
#SPMIopt
mdp.set_initial_configuration(original_model)
spmi.spmi_opt(initial_policy, initial_model)

opt_iterations = np.array(range(spmi.iteration))
opt_evaluations = np.array(spmi.evaluations)
opt_p_advantages = np.array(spmi.p_advantages)
opt_m_advantages = np.array(spmi.m_advantages)
opt_p_dist_sup = np.array(spmi.p_dist_sup)
opt_p_dist_mean = np.array(spmi.p_dist_mean)
opt_m_dist_sup = np.array(spmi.m_dist_sup)
opt_m_dist_mean = np.array(spmi.m_dist_mean)
opt_alfas = np.array(spmi.alfas)
opt_betas = np.array(spmi.betas)
opt_p_change = np.cumsum(1 - np.array(spmi.p_change))
opt_m_change = np.cumsum(1 - np.array(spmi.m_change))
opt_coefficient = np.array(spmi.w_current)
opt_bound = np.array(spmi.bound)
spmi.save_simulation_opt(dir_path, 'opt.csv')

#-------------------------------------------------------------------------------
#plots

plt.switch_backend('pdf')

plt.figure()
plt.title('Performance')
plt.xlabel('Iteration')
plt.plot(iterations, evaluations, color='b', label='SPMI')
plt.plot(sup_iterations, sup_evaluations, color='y', label='SPMI sup')
plt.plot(alt_iterations, alt_evaluations, color='g', label='alternated')
plt.plot(seq_iterations, seq_evaluations, color='r', label='sequential')
plt.plot(opt_iterations, opt_evaluations, color='m', label='optimal')
plt.plot(ntt_iterations, ntt_evaluations, color='c', label='no target trick')
plt.plot(int_iterations, int_evaluations, color='k', linestyle='dotted', label='no full step')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/performance comparison")

plt.figure()
plt.title('Model coefficient')
plt.xlabel('Iteration')
plt.plot(iterations, coefficient[:, 0], color='b', label='SPMI')
plt.plot(sup_iterations, sup_coefficient[:, 0], color='y', label='SPMI sup')
plt.plot(alt_iterations, alt_coefficient[:, 0], color='g', label='alternated')
plt.plot(seq_iterations, seq_coefficient[:, 0], color='r', label='sequential')
plt.plot(opt_iterations, opt_coefficient[:, 0], color='m', label='optimal')
plt.plot(ntt_iterations, ntt_coefficient[:, 0], color='c', label='no target trick')
plt.plot(int_iterations, int_coefficient[:, 0], color='k', linestyle='dotted', label='no full step')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/model comparison")

plt.figure()
plt.title('Bound comparison')
plt.xlabel('Iteration')
plt.plot(iterations, bound, color='b', label='SPMI')
plt.plot(sup_iterations, sup_bound, color='y', label='SPMI sup')
plt.plot(alt_iterations, alt_bound, color='g', label='alternated')
plt.plot(seq_iterations, seq_bound, color='r', label='sequential')
plt.plot(opt_iterations, opt_bound, color='m', label='optimal')
plt.plot(ntt_iterations, ntt_bound, color='c', label='no target trick')
plt.plot(int_iterations, int_bound, color='k', linestyle='dotted', label='no full step')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/bound comparison")