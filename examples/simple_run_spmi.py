import matplotlib.pyplot as plt
import numpy as np
import time

from ifqi.utils.uniform_policy import UniformPolicy
from ifqi.envs.race_track_configurable import RaceTrackConfigurableEnv
from ifqi.algorithms.spmi_exact import SPMI

path_name = "/Users/mirco/Desktop/Simulazioni"
path_file = "/Simulazione_SPMI"


startTime = time.time()


k = 0.5
mdp = RaceTrackConfigurableEnv(track_file='track0', initial_configuration=k)


print('nS: {0}'.format(mdp.nS))
print('MDP instantiated')


eps = 0.0001
spmi = SPMI(mdp, eps)
model = np.array([k, 1 - k])
policy = UniformPolicy(mdp)


policy, model = spmi.safe_policy_model_iteration(policy, model)


iterations = np.array(range(spmi.iteration))
evaluations = np.array(spmi.evaluations)
p_advantages = np.array(spmi.p_advantages)
m_advantages = np.array(spmi.m_advantages)
p_dist_sup = np.array(spmi.p_dist_sup)
m_dist_sup = np.array(spmi.m_dist_sup)
alfas = np.array(spmi.alfas)
betas = np.array(spmi.betas)
coefficient = np.array(spmi.coefficients)

plt.switch_backend('pdf')

plt.figure()
plt.title("Performance")
plt.xlabel('Iteration')
plt.plot(iterations, evaluations, label='J_p_m')
plt.legend(loc='best', fancybox=True)
plt.savefig(path_name + path_file + "/performance")

plt.figure()
plt.title('Infinite norm distances')
plt.xlabel('Iteration')
plt.plot(iterations, p_dist_sup, color='tab:blue', label='policy_distance')
plt.plot(iterations, m_dist_sup, color='tab:red', label='model_distance')
plt.legend(loc='best', fancybox=True)
plt.savefig(path_name + path_file + "/distance")

plt.figure()
plt.title('Advantages and coefficients')
plt.xlabel('Iteration')
plt.plot(iterations, p_advantages, color='tab:blue', label='policy_advantage')
plt.plot(iterations, alfas, color='tab:blue', linestyle='dashed', label='alfa')
plt.plot(iterations, m_advantages, color='tab:red', label='model_advantage')
plt.plot(iterations, betas, color='tab:red', linestyle='dashed', label='beta')
plt.legend(loc='best', fancybox=True)
plt.yscale('log')
plt.savefig(path_name + path_file + "/betas_advantage")

plt.figure()
plt.title('Model combination coefficient')
plt.xlabel('Iteration')
plt.plot(iterations, coefficient, label='coefficient')
plt.legend(loc='best', fancybox=True)
plt.savefig(path_name + path_file + "/coefficient")



print('The script took {0} minutes'.format((time.time() - startTime) / 60))