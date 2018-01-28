from spmi.envs.race_track_configurable import RaceTrackConfigurableEnv
from spmi.algorithms.spmi_exact_non_par import SPMI
from spmi.utils.uniform_policy import UniformPolicy
from spmi.algorithms.policy_chooser import *
from spmi.algorithms.model_chooser import *
import numpy as np
import matplotlib.pyplot as plt

dir_path = "/home/deep/mirco/spmi/simulations/data/simul_server_spmi_no_vertex"

mdp = RaceTrackConfigurableEnv(track_file='track0', initial_configuration=0.5)

uniform_policy = UniformPolicy(mdp)

initial_model = TabularModel(mdp.P, mdp.nS, mdp.nA)
initial_policy = TabularPolicy(uniform_policy.get_rep(), mdp.nS, mdp.nA)

model_set = [TabularModel(mdp.P1, mdp.nS, mdp.nA), TabularModel(mdp.P2, mdp.nS, mdp.nA)]

policy_chooser = GreedyPolicyChooser(mdp.nS, mdp.nA)
model_chooser = SetModelChooser(model_set, mdp.nS, mdp.nA)

eps = 0.0
spmi = SPMI(mdp, eps, policy_chooser, model_chooser, max_iter=100000, use_target_trick=True, delta_q=1)


#-------------------------------------------------------------------------------
#SPMI
policy, model = spmi.spmi(initial_policy, initial_model)

spmi.save_simulation(dir_path, "execution data")


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
coefficients = np.array(spmi.w_current)


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
plt.plot(iterations, p_dist_sup, color='b', label='policy_sup')
plt.plot(iterations, p_dist_mean, color='b', linestyle='dashed', label='policy_mean')
plt.plot(iterations, m_dist_sup, color='r', label='model_sup')
plt.plot(iterations, m_dist_mean, color='r', linestyle='dashed', label='model_mean')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/distance")

plt.figure()
plt.title('Alfa and policy advantage')
plt.xlabel('Iteration')
plt.plot(iterations, p_advantages, color='b', label='policy_advantage')
plt.plot(iterations, alfas, color='b', linestyle='dashed', label='alfa')
plt.legend(loc='best', fancybox=True)
plt.yscale('log')
plt.savefig(dir_path + "/alfa_advantage")

plt.figure()
plt.title('Beta and model advantage')
plt.xlabel('Iteration')
plt.plot(iterations, m_advantages, color='r', label='model_advantage')
plt.plot(iterations, betas, color='r', linestyle='dashed', label='beta')
plt.legend(loc='best', fancybox=True)
plt.yscale('log')
plt.savefig(dir_path + "/beta_advantage")

plt.figure()
plt.title('Coefficient')
plt.xlabel('Iteration')
plt.plot(iterations, coefficients, color='r', label='coefficient')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/coefficient")

plt.figure()
plt.title('Model Target Change')
plt.xlabel('Iteration')
plt.plot(iterations, p_change, color='r', label='target change')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/model_target_change")

plt.figure()
plt.title('Policy Target Change')
plt.xlabel('Iteration')
plt.plot(iterations, m_change, color='b', label='target change')
plt.legend(loc='best', fancybox=True)
plt.savefig(dir_path + "/policy_target_change")