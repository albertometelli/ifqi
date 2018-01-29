import matplotlib.pyplot as plt
import numpy as np
import time

from spmi.utils.uniform_policy import UniformPolicy
from spmi.envs.race_track_configurable_2 import RaceTrackConfigurableEnv
from spmi.algorithms.spmi_exact_par import SPMI

path_name = "/Users/mirco/Desktop/Simulazioni"
path_file = "/Simulazione_SPMI"


startTime = time.time()


k = 0.5
mdp = RaceTrackConfigurableEnv(track_file='track0wall', initial_configuration=k)


print('nS: {0}'.format(mdp.nS))
print('MDP instantiated')


eps = 0.00015
spmi = SPMI(mdp, eps)
initial_model = np.array([k, 1 - k])
initial_policy = UniformPolicy(mdp)



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
coefficient = np.array(spmi.coefficients)

mdp.model_configuration(k)


spmi.spmi_no_full_step(initial_policy, initial_model)

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
int_coefficient = np.array(spmi.coefficients)

mdp.model_configuration(k)


spmi.spmi_alternated(initial_policy, initial_model)

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
alt_coefficient = np.array(spmi.coefficients)

mdp.model_configuration(k)


spmi.spmi_sequential(initial_policy, initial_model)

n_spi = len(spmi.alfas)
a = n_spi
n = spmi.iteration

seq_iterations = np.array(range(n))
seq_evaluations = np.array(spmi.evaluations)
seq_p_advantages = spmi.p_advantages
seq_m_advantages = spmi.m_advantages
seq_p_dist_sup = spmi.p_dist_sup
seq_p_dist_mean = spmi.p_dist_mean
seq_m_dist_sup = spmi.m_dist_sup
seq_m_dist_mean = spmi.m_dist_mean
seq_alfas = spmi.alfas
seq_betas = spmi.betas
seq_coefficient = spmi.coefficients



plt.switch_backend('pdf')

plt.figure()
plt.title('Performance')
plt.xlabel('Iteration')
plt.plot(iterations, evaluations, color='b', label='SPMI')
plt.plot(alt_iterations, alt_evaluations, color='g', label='alternated')
plt.plot(seq_iterations, seq_evaluations, color='r', label='sequential')
plt.plot(int_iterations, int_evaluations, color='k', linestyle='dotted', label='intertwined')
plt.legend(loc='best', fancybox=True)
plt.savefig(path_name + path_file + "/performance")

plt.figure()
plt.title('Policy distances')
plt.xlabel('Iteration')
plt.plot(iterations, p_dist_sup, color='b', label='SPMI sup')
plt.plot(iterations, p_dist_mean, color='b', linestyle='dashed', label='SPMI mean')
plt.plot(alt_iterations, alt_p_dist_sup, color='g', label='alternated sup')
plt.plot(alt_iterations, alt_p_dist_mean, color='g', linestyle='dashed', label='alternated mean')
plt.plot(seq_iterations, seq_p_dist_sup, color='r', label='sequential sup')
plt.plot(seq_iterations, seq_p_dist_mean, color='r', linestyle='dashed', label='sequential mean')
plt.plot(int_iterations, int_p_dist_sup, color='k', linestyle='dotted', label='intertwined sup')
plt.plot(int_iterations, int_p_dist_mean, color='k', linestyle='dashed', label='intertwined mean')
plt.legend(loc='best', fancybox=True)
plt.savefig(path_name + path_file + "/policy distance")

plt.figure()
plt.title('Model distances')
plt.xlabel('Iteration')
plt.plot(iterations, m_dist_sup, color='b', label='SPMI sup')
plt.plot(iterations, m_dist_mean, color='b', linestyle='dashed', label='SPMI mean')
plt.plot(alt_iterations, alt_m_dist_sup, color='g', label='alternated sup')
plt.plot(alt_iterations, alt_m_dist_mean, color='g', linestyle='dashed', label='alternated mean')
plt.plot(seq_iterations, seq_m_dist_sup, color='r', label='sequential sup')
plt.plot(seq_iterations, seq_m_dist_mean, color='r', linestyle='dashed', label='sequential mean')
plt.plot(int_iterations, int_m_dist_sup, color='k', linestyle='dotted', label='intertwined sup')
plt.plot(int_iterations, int_m_dist_mean, color='k', linestyle='dashed', label='intertwined mean')
plt.legend(loc='best', fancybox=True)
plt.savefig(path_name + path_file + "/model distance")

plt.figure()
plt.title('Advantage')
plt.xlabel('Iteration')
plt.plot(iterations, p_advantages, color='b', label='SPMI policy')
plt.plot(iterations, m_advantages, color='b', linestyle='dashed', label='SPMI model')
plt.plot(alt_iterations, alt_p_advantages, color='g', label='alternated policy')
plt.plot(alt_iterations, alt_m_advantages, color='g', linestyle='dashed', label='alternated model')
plt.plot(seq_iterations, seq_p_advantages, color='r', label='sequential policy')
plt.plot(seq_iterations, seq_m_advantages, color='r', linestyle='dashed', label='sequential model')
plt.plot(int_iterations, int_p_advantages, color='k', linestyle='dotted', label='intertwined policy')
plt.plot(int_iterations, int_m_advantages, color='k', linestyle='dashed', label='intertwined model')
plt.legend(loc='best', fancybox=True)
plt.savefig(path_name + path_file + "/advantages")

plt.figure()
plt.title('Alfa & Beta')
plt.xlabel('Iteration')
plt.plot(iterations, alfas, color='b', linestyle='', marker='o', label='SPMI alfa')
plt.plot(iterations, betas, color='b', linestyle='', marker='s', label='SPMI beta')
plt.plot(alt_iterations, alt_alfas, color='g', linestyle='', marker='o', label='alternated alfa')
plt.plot(alt_iterations, alt_betas, color='g', linestyle='', marker='s', label='alternated beta')
plt.plot(seq_iterations, seq_alfas, color='r', linestyle='', marker='o', label='sequential alfa')
plt.plot(seq_iterations, seq_betas, color='r', linestyle='', marker='s', label='sequential beta')
plt.plot(int_iterations, int_alfas, color='k', linestyle='', marker='o', label='intertwined alfa')
plt.plot(int_iterations, int_betas, color='k', linestyle='', marker='s', label='intertwined beta')
plt.legend(loc='best', fancybox=True)
plt.yscale('log')
plt.savefig(path_name + path_file + "/alfabeta")

plt.figure()
plt.title('Model combination coefficient')
plt.xlabel('Iteration')
plt.plot(iterations, coefficient, color='b', label='SPMI')
plt.plot(alt_iterations, alt_coefficient, color='g', label='alternated')
plt.plot(seq_iterations, seq_coefficient, color='r', linestyle='dashdot', label='sequential')
plt.plot(int_iterations, int_coefficient, color='k', linestyle='dotted', label='intertwined')
plt.legend(loc='best', fancybox=True)
plt.savefig(path_name + path_file + "/coefficient")


print('The script took {0} minutes'.format((time.time() - startTime) / 60))