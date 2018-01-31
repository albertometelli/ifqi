from spmi.envs.race_track_configurable_4 import RaceTrackConfigurableEnv
from spmi.utils.uniform_policy import UniformPolicy
from spmi.algorithms.spmi_exact_non_par import SPMI
from spmi.algorithms.policy_chooser import *
from spmi.algorithms.model_chooser import *
import numpy as np
import matplotlib.pyplot as plt
import copy
import os


track = 'race_straight'
simulation_name = 'simul_4_opt'
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
#SPMI sup
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
opt_w_target = np.array(spmi.w_target)
opt_bound = np.array(spmi.bound)
spmi.save_simulation_opt(dir_path, 'opt.csv')

#-------------------------------------------------------------------------------
#plots

plt.switch_backend('pdf')

plt.figure()
plt.title('Performance')
plt.xlabel('Iteration')
plt.plot(iterations, evaluations, color='b', label='J spmi')
plt.plot(opt_iterations, opt_evaluations, color='r', linestyle='dashed', label='J spmi_opt')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/performance comparison")

plt.figure()
plt.title('Beta')
plt.xlabel('Iteration')
plt.plot(iterations, betas, color='b', linestyle='dotted', label='beta spmi')
plt.plot(opt_iterations, opt_betas, color='r', linestyle='dotted', label='beta spmi_opt')
plt.legend(loc='best', fancybox=True)
plt.yscale('log')
plt.savefig(dir_path + "/beta comparison")

plt.figure()
plt.title('Alfa')
plt.xlabel('Iteration')
plt.plot(iterations, alfas, color='b', linestyle='dotted', label='alfa spmi')
plt.plot(opt_iterations, opt_alfas, color='r', linestyle='dotted', label='alfa spmi_opt')
plt.legend(loc='best', fancybox=True)
plt.yscale('log')
plt.savefig(dir_path + "/alfa comparison")

plt.figure()
plt.title('P_hs_nb coefficient')
plt.xlabel('Iteration')
plt.plot(iterations, coefficient[:, 0], color='b', label='SPMI')
plt.plot(opt_iterations, opt_coefficient[:, 0], color='r', linestyle='dotted', label='spmi_opt')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/P_hs_nb")

plt.figure()
plt.title('P_ls_nb coefficient')
plt.xlabel('Iteration')
plt.plot(iterations, coefficient[:, 1], color='b', label='SPMI')
plt.plot(opt_iterations, opt_coefficient[:, 1], color='r', linestyle='dotted', label='spmi_opt')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/P_ls_nb")

plt.figure()
plt.title('P_hs_b coefficient')
plt.xlabel('Iteration')
plt.plot(iterations, coefficient[:, 2], color='b', label='SPMI')
plt.plot(opt_iterations, opt_coefficient[:, 2], color='r', linestyle='dotted', label='spmi_opt')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/P_hs_b")

plt.figure()
plt.title('P_ls_b coefficient')
plt.xlabel('Iteration')
plt.plot(iterations, coefficient[:, 3], color='b', label='SPMI notarget')
plt.plot(opt_iterations, opt_coefficient[:, 3], color='r', linestyle='dotted', label='spmi_opt')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/P_ls_b")

plt.figure()
plt.title('Target coefficients')
plt.xlabel('Iteration')
plt.plot(opt_iterations, opt_coefficient[:, 0], color='b', label='spmi_opt_tt')
plt.plot(opt_iterations, opt_w_target[:, 0], color='r', linestyle='dotted', label='spmi_opt')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/target comparison")

plt.figure()
plt.title('Bound comparison')
plt.xlabel('Iteration')
plt.plot(iterations, bound, color='b', linestyle='dashed', label='spmi')
plt.plot(opt_iterations, opt_bound, color='r', linestyle='dotted', label='spmi_opt')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/bound comparison")