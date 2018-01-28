from spmi.envs.race_track_configurable import RaceTrackConfigurableEnv
from spmi.utils.uniform_policy import UniformPolicy
from spmi.algorithms.spmi_exact_non_par import SPMI
from spmi.algorithms.policy_chooser import *
from spmi.algorithms.model_chooser import *
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import os


startTime = time.time()

track = 'race_2start'
simulation_name = 'simul_' + track
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
spmi = SPMI(mdp, eps, policy_chooser, model_chooser, max_iter=30000, use_target_trick=True, delta_q=1)

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
spmi.save_simulation(dir_path, 'spmi.csv')

#-------------------------------------------------------------------------------
#SPMI sup
mdp.set_model(original_model)
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
spmi.save_simulation(dir_path, 'sup.csv')

#-------------------------------------------------------------------------------
#SPMI no full step
mdp.set_model(original_model)
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
spmi.save_simulation(dir_path,  'nofull.csv')

#-------------------------------------------------------------------------------
#SPMI alternated
mdp.set_model(original_model)
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
spmi.save_simulation(dir_path, 'alt.csv')

#-------------------------------------------------------------------------------
#SPMI sequential
mdp.set_model(original_model)
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
spmi.save_simulation(dir_path, 'seq.csv')

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
plt.plot(int_iterations, int_evaluations, color='k', linestyle='dotted', label='no full step')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/performance")

plt.figure()
plt.title('Policy distances')
plt.xlabel('Iteration')
plt.plot(iterations, p_dist_sup, color='b', label='SPMI sup')
plt.plot(iterations, p_dist_mean, color='b', linestyle='dashed', label='SPMI mean')
plt.plot(sup_iterations, sup_p_dist_sup, color='y', label='sup sup')
plt.plot(sup_iterations, sup_p_dist_mean, color='y', linestyle='dashed', label='sup mean')
plt.plot(alt_iterations, alt_p_dist_sup, color='g', label='alternated sup')
plt.plot(alt_iterations, alt_p_dist_mean, color='g', linestyle='dashed', label='alternated mean')
plt.plot(seq_iterations, seq_p_dist_sup, color='r', label='sequential sup')
plt.plot(seq_iterations, seq_p_dist_mean, color='r', linestyle='dashed', label='sequential mean')
plt.plot(int_iterations, int_p_dist_sup, color='k', linestyle='dotted', label='no full step sup')
plt.plot(int_iterations, int_p_dist_mean, color='k', linestyle='dashed', label='no full step mean')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/policy distance")

plt.figure()
plt.title('Model distances')
plt.xlabel('Iteration')
plt.plot(iterations, m_dist_sup, color='b', label='SPMI sup')
plt.plot(iterations, m_dist_mean, color='b', linestyle='dashed', label='SPMI mean')
plt.plot(sup_iterations, sup_m_dist_sup, color='y', label='sup sup')
plt.plot(sup_iterations, sup_m_dist_mean, color='y', linestyle='dashed', label='sup mean')
plt.plot(alt_iterations, alt_m_dist_sup, color='g', label='alternated sup')
plt.plot(alt_iterations, alt_m_dist_mean, color='g', linestyle='dashed', label='alternated mean')
plt.plot(seq_iterations, seq_m_dist_sup, color='r', label='sequential sup')
plt.plot(seq_iterations, seq_m_dist_mean, color='r', linestyle='dashed', label='sequential mean')
plt.plot(int_iterations, int_m_dist_sup, color='k', linestyle='dotted', label='np full step sup')
plt.plot(int_iterations, int_m_dist_mean, color='k', linestyle='dashed', label='no full step mean')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/model distance")

plt.figure()
plt.title('Advantage')
plt.xlabel('Iteration')
plt.plot(iterations, p_advantages, color='b', label='SPMI policy')
plt.plot(iterations, m_advantages, color='b', linestyle='dashed', label='SPMI model')
plt.plot(sup_iterations, sup_p_advantages, color='y', label='sup policy')
plt.plot(sup_iterations, sup_m_advantages, color='y', linestyle='dashed', label='sup model')
plt.plot(alt_iterations, alt_p_advantages, color='g', label='alternated policy')
plt.plot(alt_iterations, alt_m_advantages, color='g', linestyle='dashed', label='alternated model')
plt.plot(seq_iterations, seq_p_advantages, color='r', label='sequential policy')
plt.plot(seq_iterations, seq_m_advantages, color='r', linestyle='dashed', label='sequential model')
plt.plot(int_iterations, int_p_advantages, color='k', linestyle='dotted', label='no full step policy')
plt.plot(int_iterations, int_m_advantages, color='k', linestyle='dashed', label='no full step model')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/advantages")

plt.figure()
plt.title('Alfa & Beta')
plt.xlabel('Iteration')
plt.plot(iterations, alfas, color='b', linestyle='', marker='o', label='SPMI alfa')
plt.plot(iterations, betas, color='b', linestyle='', marker='s', label='SPMI beta')
plt.plot(sup_iterations, sup_alfas, color='y', linestyle='', marker='o', label='sup alfa')
plt.plot(sup_iterations, sup_betas, color='y', linestyle='', marker='s', label='sup beta')
plt.plot(alt_iterations, alt_alfas, color='g', linestyle='', marker='o', label='alternated alfa')
plt.plot(alt_iterations, alt_betas, color='g', linestyle='', marker='s', label='alternated beta')
plt.plot(seq_iterations, seq_alfas, color='r', linestyle='', marker='o', label='sequential alfa')
plt.plot(seq_iterations, seq_betas, color='r', linestyle='', marker='s', label='sequential beta')
plt.plot(int_iterations, int_alfas, color='k', linestyle='', marker='o', label='no full step alfa')
plt.plot(int_iterations, int_betas, color='k', linestyle='', marker='s', label='no full step beta')
plt.legend(loc='best', fancybox=True)
plt.yscale('log')
plt.savefig(dir_path + "/alfabeta")

plt.figure()
plt.title('Policy Model Changes')
plt.xlabel('Iteration')
plt.plot(iterations, p_change, color='b', label='SPMI policy')
plt.plot(iterations, m_change, color='b', linestyle='dashed', label='SPMI model')
plt.plot(sup_iterations, sup_p_change, color='y', label='sup policy')
plt.plot(sup_iterations, sup_m_change, color='y', linestyle='dashed', label='sup model')
plt.plot(alt_iterations, alt_p_change, color='g', label='alternated policy')
plt.plot(alt_iterations, alt_m_change, color='g', linestyle='dashed', label='alternated model')
plt.plot(seq_iterations, seq_p_change, color='r', label='sequential policy')
plt.plot(seq_iterations, seq_m_change, color='r', linestyle='dashed', label='sequential model')
plt.plot(int_iterations, int_p_change, color='k', linestyle='dotted', label='no full step policy')
plt.plot(int_iterations, int_m_change, color='k', linestyle='dashed', label='no full step model')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/changes")

plt.figure()
plt.title('Model coefficient')
plt.xlabel('Iteration')
plt.plot(iterations, coefficient, color='b', label='SPMI')
plt.plot(sup_iterations, sup_coefficient, color='y', label='SPMI sup')
plt.plot(alt_iterations, alt_coefficient, color='g', label='alternated')
plt.plot(seq_iterations, seq_coefficient, color='r', label='sequential')
plt.plot(int_iterations, int_coefficient, color='k', linestyle='dotted', label='no full step')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/coefficient")



print('The script took {0} minutes'.format((time.time() - startTime) / 60))