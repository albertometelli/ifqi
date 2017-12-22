import matplotlib.pyplot as plt
import numpy as np
import time

from spmi.utils.uniform_policy import UniformPolicy
from spmi.envs.race_track_configurable import RaceTrackConfigurableEnv
from spmi.algorithms.spmi_exact import SPMI

path_name = "/Users/mirco/Desktop/Simulazioni"
path_file = "/Simulazione_SPMI"


startTime = time.time()


k = 0.5
mdp = RaceTrackConfigurableEnv(track_file='track_spirale', initial_configuration=k)


print('nS: {0}'.format(mdp.nS))
print('MDP instantiated')


eps = 0.00001
spmi = SPMI(mdp, eps)
model = np.array([k, 1 - k])
policy = UniformPolicy(mdp)


policy, model = spmi.safe_policy_model_iteration(policy, model)


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

plt.switch_backend('pdf')

plt.figure()
plt.title('Performance')
plt.xlabel('Iteration')
plt.plot(iterations, evaluations, label='J_p_m')
plt.legend(loc='best', fancybox=True)
plt.savefig(path_name + path_file + "/performance")

plt.figure()
plt.title('Policy and model distances')
plt.xlabel('Iteration')
plt.plot(iterations, p_dist_sup, color='tab:blue', label='policy_sup')
plt.plot(iterations, p_dist_mean, color='tab:blue', linestyle='dashed', label='policy_mean')
plt.plot(iterations, m_dist_sup, color='tab:red', label='model_sup')
plt.plot(iterations, m_dist_mean, color='tab:red', linestyle='dashed', label='model_mean')
plt.legend(loc='best', fancybox=True)
plt.savefig(path_name + path_file + "/distance")

plt.figure()
plt.title('Alfa and policy advantage')
plt.xlabel('Iteration')
plt.plot(iterations, p_advantages, color='tab:blue', label='policy_advantage')
plt.plot(iterations, alfas, color='tab:blue', linestyle='dashed', label='alfa')
plt.legend(loc='best', fancybox=True)
plt.yscale('log')
plt.savefig(path_name + path_file + "/alfa_advantage")

plt.figure()
plt.title('Beta and model advantage')
plt.xlabel('Iteration')
plt.plot(iterations, m_advantages, color='tab:red', label='model_advantage')
plt.plot(iterations, betas, color='tab:red', linestyle='dashed', label='beta')
plt.legend(loc='best', fancybox=True)
plt.yscale('log')
plt.savefig(path_name + path_file + "/beta_advantage")

plt.figure()
plt.title('Model combination coefficient')
plt.xlabel('Iteration')
plt.plot(iterations, coefficient, color='tab:purple', label='coefficient')
plt.legend(loc='best', fancybox=True)
plt.savefig(path_name + path_file + "/coefficient")



print('The script took {0} minutes'.format((time.time() - startTime) / 60))