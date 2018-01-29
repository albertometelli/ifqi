from spmi.envs.race_track_configurable_4 import RaceTrackConfigurableEnv
from spmi.algorithms.spmi_exact_non_par import SPMI
from spmi.utils.uniform_policy import UniformPolicy
from spmi.algorithms.policy_chooser import *
from spmi.algorithms.model_chooser import *
import numpy as np
import matplotlib.pyplot as plt
import copy

dir_path = "/Users/mirco/Desktop/Simulazioni/optSPMI"

mdp = RaceTrackConfigurableEnv(track_file='race_straight', initial_configuration=[0.5, 0.5, 0, 0], pfail=0.07)

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
spmi = SPMI(mdp, eps, policy_chooser, model_chooser, max_iter=1000, use_target_trick=True, delta_q=1)


#-------------------------------------------------------------------------------
#SPMI_opt

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
opt_x = np.array(spmi.x)
opt_p_change = np.cumsum(1 - np.array(spmi.p_change))
opt_m_change = np.cumsum(1 - np.array(spmi.m_change))
opt_coefficient = np.array(spmi.w_current)
opt_w_target = np.array(spmi.w_target)
opt_bound = np.array(spmi.bound)


#-------------------------------------------------------------------------------
#SPMI_opt_tt

mdp.set_initial_configuration(original_model)
spmi.optimal_spmi(initial_policy, initial_model)

opt_ntt_iterations = np.array(range(spmi.iteration))
opt_ntt_evaluations = np.array(spmi.evaluations)
opt_ntt_p_advantages = np.array(spmi.p_advantages)
opt_ntt_m_advantages = np.array(spmi.m_advantages)
opt_ntt_p_dist_sup = np.array(spmi.p_dist_sup)
opt_ntt_p_dist_mean = np.array(spmi.p_dist_mean)
opt_ntt_m_dist_sup = np.array(spmi.m_dist_sup)
opt_ntt_m_dist_mean = np.array(spmi.m_dist_mean)
opt_ntt_alfas = np.array(spmi.alfas)
opt_ntt_betas = np.array(spmi.betas)
opt_ntt_x = np.array(spmi.x)
opt_ntt_p_change = np.cumsum(1 - np.array(spmi.p_change))
opt_ntt_m_change = np.cumsum(1 - np.array(spmi.m_change))
opt_ntt_coefficient = np.array(spmi.w_current)
opt_ntt_w_target = np.array(spmi.w_target)
opt_ntt_bound = np.array(spmi.bound)


#-------------------------------------------------------------------------------
#SPMI

mdp.set_initial_configuration(original_model)
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


#-------------------------------------------------------------------------------
#SPMI_no_tt

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


plt.switch_backend('pdf')

plt.figure()
plt.title('Performance')
plt.xlabel('Iteration')
plt.plot(iterations, evaluations, color='tab:red', label='J spmi')
plt.plot(ntt_iterations, ntt_evaluations, color='tab:orange', label='J spmi_ntt')
plt.plot(opt_iterations, opt_evaluations, color='tab:blue', linestyle='dashed', label='J spmi_opt_tt')
plt.plot(opt_ntt_iterations, opt_ntt_evaluations, color='tab:cyan', linestyle='dashed', label='J spmi_opt')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/performance comparison")

plt.figure()
plt.title('Beta')
plt.xlabel('Iteration')
plt.plot(iterations, betas, color='tab:red', linestyle='dotted', label='beta spmi')
plt.plot(ntt_iterations, ntt_betas, color='tab:orange', linestyle='dotted', label='beta spmi_ntt')
plt.plot(opt_iterations, opt_betas, color='tab:blue', linestyle='dotted', label='beta spmi_opt_tt')
plt.plot(opt_ntt_iterations, opt_ntt_betas, color='tab:cyan', linestyle='dotted', label='beta spmi_opt')
plt.legend(loc='best', fancybox=True)
plt.yscale('log')
plt.savefig(dir_path + "/beta comparison")

plt.figure()
plt.title('Alfa')
plt.xlabel('Iteration')
plt.plot(iterations, alfas, color='tab:red', linestyle='dotted', label='alfa spmi')
plt.plot(ntt_iterations, ntt_alfas, color='tab:orange', linestyle='dotted', label='alfa spmi_ntt')
plt.plot(opt_iterations, opt_alfas, color='tab:blue', linestyle='dotted', label='alfa spmi_opt_tt')
plt.plot(opt_ntt_iterations, opt_ntt_alfas, color='tab:cyan', linestyle='dotted', label='alfa spmi_opt')
plt.legend(loc='best', fancybox=True)
plt.yscale('log')
plt.savefig(dir_path + "/alfa comparison")

plt.figure()
plt.title('Model coefficient')
plt.xlabel('Iteration')
plt.plot(iterations, coefficient[:, 0], color='tab:red', linestyle='dashed', label='spmi')
plt.plot(ntt_iterations, ntt_coefficient[:, 0], color='tab:orange', linestyle='dashed', label='spmi_ntt')
plt.plot(opt_iterations, opt_coefficient[:, 0], color='tab:blue', linestyle='dotted', label='spmi_opt_tt')
plt.plot(opt_ntt_iterations, opt_ntt_coefficient[:, 0], color='tab:cyan', linestyle='dotted', label='spmi_opt')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/model comparison")

plt.figure()
plt.title('Target models')
plt.xlabel('Iteration')
plt.plot(opt_iterations, opt_coefficient[:, 0], color='tab:blue', label='spmi_opt_tt')
plt.plot(opt_ntt_iterations, opt_ntt_coefficient[:, 0], color='tab:cyan', label='spmi_opt')
plt.plot(opt_iterations, opt_w_target[:, 0], color='tab:blue', linestyle='dotted', label='spmi_opt_tt')
plt.plot(opt_ntt_iterations, opt_ntt_w_target[:, 0], color='tab:cyan', linestyle='dotted', label='spmi_opt')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/target comparison")

plt.figure()
plt.title('Target/current spmi_opt_tt')
plt.xlabel('Iteration')
plt.plot(opt_iterations, opt_coefficient[:, 0], color='tab:blue', label='current')
plt.plot(opt_iterations, opt_w_target[:, 0], color='tab:blue', linestyle='dotted', label='target')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/target current opt_tt")

plt.figure()
plt.title('Target/current spmi_opt_ntt')
plt.xlabel('Iteration')
plt.plot(opt_ntt_iterations, opt_ntt_coefficient[:, 0], color='tab:cyan', label='current')
plt.plot(opt_ntt_iterations, opt_ntt_w_target[:, 0], color='tab:cyan', linestyle='dotted', label='target')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/target current opt_ntt")

plt.figure()
plt.title('Bound comparison')
plt.xlabel('Iteration')
plt.plot(iterations, bound, color='tab:red', linestyle='dashed', label='spmi')
plt.plot(ntt_iterations, ntt_bound, color='tab:orange', linestyle='dashed', label='spmi_ntt')
plt.plot(opt_iterations, opt_bound, color='tab:blue', linestyle='dotted', label='spmi_opt_tt')
plt.plot(opt_ntt_iterations, opt_ntt_bound, color='tab:cyan', linestyle='dotted', label='spmi_opt')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/bound comparison")