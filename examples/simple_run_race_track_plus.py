from spmi.envs.race_track_configurable_plus import RaceTrackConfigurableEnv
from spmi.algorithms.spmi_exact_non_par import SPMI
from spmi.utils.uniform_policy import UniformPolicy
from spmi.algorithms.policy_chooser import *
from spmi.algorithms.model_chooser import *
import numpy as np
import matplotlib.pyplot as plt

dir_path = "/Users/mirco/Desktop/Simulazioni/SPMI_non_par"

mdp = RaceTrackConfigurableEnv(track_file='track0_start', initial_configuration=0.5)

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
spmi = SPMI(mdp, eps, policy_chooser, model_chooser, max_iter=50000, use_target_trick=True, delta_q=1)


#-------------------------------------------------------------------------------
#SPMI
policy, model = spmi.spmi(initial_policy, initial_model)

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



# # coefficient computation and print
# P = mdp.P_sa
# P1 = mdp.P1_sa
# P2 = mdp.P2_sa
# k = (P - P2) / (P1 - P2 + 1e-24)
# k = np.max(k)
# print('\n---- current k: {0} ----'.format(k))


plt.switch_backend('pdf')

plt.figure()
plt.title('Performance')
plt.xlabel('Iteration')
plt.plot(iterations, evaluations, label='J_p_m')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/performance")

plt.figure()
plt.title('Policy and model distances')
plt.xlabel('Iteration')
plt.plot(iterations, p_dist_sup, color='tab:blue', label='policy_sup')
plt.plot(iterations, p_dist_mean, color='tab:blue', linestyle='dashed', label='policy_mean')
plt.plot(iterations, m_dist_sup, color='tab:red', label='model_sup')
plt.plot(iterations, m_dist_mean, color='tab:red', linestyle='dashed', label='model_mean')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/distance")

plt.figure()
plt.title('Alfa and policy advantage')
plt.xlabel('Iteration')
plt.plot(iterations, p_advantages, color='tab:blue', label='policy_advantage')
plt.plot(iterations, alfas, color='tab:blue', linestyle='dashed', label='alfa')
plt.legend(loc='best', fancybox=True)
plt.yscale('log')
plt.savefig(dir_path + "/alfa_advantage")

plt.figure()
plt.title('Beta and model advantage')
plt.xlabel('Iteration')
plt.plot(iterations, m_advantages, color='tab:red', label='model_advantage')
plt.plot(iterations, betas, color='tab:red', linestyle='dashed', label='beta')
plt.legend(loc='best', fancybox=True)
plt.yscale('log')
plt.savefig(dir_path + "/beta_advantage")