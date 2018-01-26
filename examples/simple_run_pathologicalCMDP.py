from spmi.envs.pathologicalCMDP import PathologicalCMDP
from spmi.algorithms.spmi_exact_non_par import SPMI
from spmi.utils.uniform_policy import UniformPolicy
from spmi.algorithms.policy_chooser import *
from spmi.algorithms.model_chooser import *
import numpy as np
import matplotlib.pyplot as plt
import copy

dir_path = "/Users/mirco/Desktop/Simulazioni/optSPMI_pathologicalCMDP"

mdp = PathologicalCMDP(p=0.1, w=0.5, M=100)

original_model = copy.deepcopy(mdp.P)

uniform_policy = UniformPolicy(mdp)

initial_model = TabularModel(mdp.P, mdp.nS, mdp.nA)
initial_policy = TabularPolicy(uniform_policy.get_rep(), mdp.nS, mdp.nA)

model_set = [TabularModel(mdp.P0, mdp.nS, mdp.nA), TabularModel(mdp.P1, mdp.nS, mdp.nA)]

policy_chooser = GreedyPolicyChooser(mdp.nS, mdp.nA)
model_chooser = SetModelChooser(model_set, mdp.nS, mdp.nA)

eps = 0.0
spmi = SPMI(mdp, eps, policy_chooser, model_chooser, max_iter=30, use_target_trick=True, delta_q=1)


#-------------------------------------------------------------------------------
#SPMI_opt

spmi.optimal_spmi(initial_policy, initial_model)

opt_iterations = np.array(range(spmi.iteration))
opt_evaluations = np.array(spmi.evaluations)
opt_p_advantages = np.array(spmi.p_advantages)
opt_m_advantages = np.array(spmi.m_advantages)
opt_p_dist_sup = np.array(spmi.p_dist_sup)
opt_p_dist_mean = np.array(spmi.p_dist_mean)
opt_m_dist_sup = np.array(spmi.m_dist_sup)
opt_m_dist_mean = np.array(spmi.m_dist_mean)
opt_alfas = np.array(spmi.alfas)
opt_x = np.array(spmi.x)
opt_betas = np.array(spmi.betas)
opt_p_change = np.cumsum(1 - np.array(spmi.p_change))
opt_m_change = np.cumsum(1 - np.array(spmi.m_change))
opt_coefficient = np.array(spmi.coefficients)
opt_w_target = np.array(spmi.w_target)
opt_bound = np.array(spmi.bound)


# plt.switch_backend('pdf')
#
# plt.figure()
# plt.title('Performance')
# plt.xlabel('Iteration')
# plt.plot(opt_iterations, opt_evaluations, label='J spmi_opt')
# plt.legend(loc='best', fancybox=True)
# plt.savefig(dir_path + "/performance opt")
#
# plt.figure()
# plt.title('X_i')
# plt.xlabel('Iteration')
# plt.plot(opt_iterations, opt_betas[:, 0], color='tab:red', linestyle='dotted', label='beta_0 spmi_opt')
# plt.plot(opt_iterations, opt_betas[:, 1], color='tab:orange', linestyle='dotted', label='beta_1 spmi_opt')
# plt.legend(loc='best', fancybox=True)
# plt.savefig(dir_path + "/x_i opt")
#
# plt.figure()
# plt.title('Model coefficient')
# plt.xlabel('Iteration')
# plt.plot(opt_iterations, opt_coefficient, color='tab:red', label='spmi_opt')
# plt.legend(loc='best', fancybox=True)
# plt.savefig(dir_path + "/model opt")


#-------------------------------------------------------------------------------
#SPMI

mdp.set_model(original_model)
spmi.safe_policy_model_iteration(initial_policy, initial_model)

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
coefficient = np.array(spmi.coefficients)
bound = np.array(spmi.bound)


plt.switch_backend('pdf')

plt.figure()
plt.title('Performance comparison')
plt.xlabel('Iteration')
plt.plot(iterations, evaluations, label='J spmi')
plt.plot(opt_iterations, opt_evaluations, linestyle='dashed', label='J spmi_opt')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/performance comparison")

plt.figure()
plt.title('Advantage 0&1')
plt.xlabel('Iteration')
plt.plot(iterations, m_advantages, color='tab:blue', linestyle='dotted', label='adv spmi')
plt.plot(opt_iterations, opt_m_advantages[:, 0], color='tab:red', linestyle='dotted', label='adv0 spmi_opt')
plt.plot(opt_iterations, opt_m_advantages[:, 1], color='tab:orange', linestyle='dotted', label='adv1 spmi_opt')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/advantage comparison 0&1")

plt.figure()
plt.title('Advantage 0')
plt.xlabel('Iteration')
plt.plot(iterations, m_advantages, color='tab:blue', linestyle='dotted', label='adv spmi')
plt.plot(opt_iterations, opt_m_advantages[:, 0], color='tab:red', linestyle='dotted', label='adv0 spmi_opt')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/advantage comparison 0")

plt.figure()
plt.title('Advantage 1')
plt.xlabel('Iteration')
plt.plot(iterations, m_advantages, color='tab:blue', linestyle='dotted', label='adv spmi')
plt.plot(opt_iterations, opt_m_advantages[:, 1], color='tab:orange', linestyle='dotted', label='adv1 spmi_opt')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/advantage comparison 1")

plt.figure()
plt.title('Beta comparison')
plt.xlabel('Iteration')
plt.plot(iterations, betas, color='tab:blue', linestyle='dotted', label='beta spmi')
plt.plot(opt_iterations, opt_betas, color='tab:orange', linestyle='dotted', label='beta spmi_opt')
plt.legend(loc='best', fancybox=True)
plt.yscale('log')
plt.savefig(dir_path + "/beta x_i comparison")

plt.figure()
plt.title('Beta and X_i')
plt.xlabel('Iteration')
plt.plot(iterations, betas, color='tab:blue', linestyle='dotted', label='beta spmi')
plt.plot(opt_iterations, opt_x[:, 0], color='tab:red', linestyle='dotted', label='x_0 spmi_opt')
plt.plot(opt_iterations, opt_x[:, 1], color='tab:orange', linestyle='dotted', label='x_1 spmi_opt')
plt.legend(loc='best', fancybox=True)
plt.yscale('log')
plt.savefig(dir_path + "/beta x_i comparison")

plt.figure()
plt.title('Beta and X_0')
plt.xlabel('Iteration')
plt.plot(iterations, betas, color='tab:blue', linestyle='dotted', label='beta spmi')
plt.plot(opt_iterations, opt_x[:, 0], color='tab:red', linestyle='dotted', label='x_0 spmi_opt')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/beta x_0 comparison")

plt.figure()
plt.title('Beta and X_1')
plt.xlabel('Iteration')
plt.plot(iterations, betas, color='tab:blue', linestyle='dotted', label='beta spmi')
plt.plot(opt_iterations, opt_x[:, 1], color='tab:orange', linestyle='dotted', label='x_1 spmi_opt')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/beta x_1 comparison")

plt.figure()
plt.title('Model coefficient')
plt.xlabel('Iteration')
plt.plot(iterations, coefficient, color='tab:blue', linestyle='dashed', label='spmi')
plt.plot(opt_iterations, opt_coefficient, color='tab:red', linestyle='dotted', label='spmi_opt')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/model comparison")

plt.figure()
plt.title('Model coefficient and Target coefficient')
plt.xlabel('Iteration')
plt.plot(opt_iterations, opt_w_target[:, 0], color='tab:red', linestyle='dashed', label='target model')
plt.plot(opt_iterations, opt_coefficient, color='tab:orange', linestyle='dotted', label='current model')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/target current comparison")

plt.figure()
plt.title('Beta comparison')
plt.xlabel('Iteration')
plt.plot(iterations, betas, color='tab:blue', linestyle='dotted', label='beta spmi')
plt.plot(opt_iterations, opt_betas, color='tab:orange', linestyle='dotted', label='beta spmi_opt')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/beta comparison")

plt.figure()
plt.title('Bound comparison')
plt.xlabel('Iteration')
plt.plot(iterations, bound, color='tab:red', linestyle='dashed', label='spmi')
plt.plot(opt_iterations, opt_bound, color='tab:blue', linestyle='dotted', label='spmi_opt')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/bound comparison")
