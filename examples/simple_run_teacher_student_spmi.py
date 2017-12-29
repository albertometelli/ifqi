from spmi.envs.teacher_student import TeacherStudentEnv
from spmi.evaluation.evaluation import collect_episodes
from spmi.algorithms.spi_exact import SPI
from spmi.utils.uniform_policy import UniformPolicy
import numpy as np
from gym.utils import seeding
import matplotlib.pyplot as plt

mdp = TeacherStudentEnv(n_literals=2,
                        max_value=1,
                        max_update=1,
                        max_literals_in_examples=2,
                        horizon=10)

uniform_policy = UniformPolicy(mdp)


dataset = collect_episodes(mdp, n_episodes=10)


eps = 0.0
delta = 0.1
spi = SPI(mdp, eps, delta, max_iter=10000)
policy = spi.safe_policy_iteration_target_trick(uniform_policy)

iterations = np.array(range(spi.iteration))
evaluations = np.array(spi.evaluations)
p_advantages = np.array(spi.advantages)
p_dist_sup = np.array(spi.distances_sup)
p_dist_mean = np.array(spi.distances_mean)
alfas = np.array(spi.alfas)

plt.switch_backend('pdf')

plt.figure()
plt.title('Performance')
plt.xlabel('Iteration')
plt.plot(iterations, evaluations, label='J_p_m')
plt.legend(loc='best', fancybox=True)
#plt.savefig(path_name + path_file + "/performance")

plt.figure()
plt.title('Policy and model distances')
plt.xlabel('Iteration')
plt.plot(iterations, p_dist_sup, color='b', label='policy_sup')
plt.plot(iterations, p_dist_mean, color='b', linestyle='dashed', label='policy_mean')
#plt.plot(iterations, m_dist_sup, color='tab:red', label='model_sup')
#plt.plot(iterations, m_dist_mean, color='tab:red', linestyle='dashed', label='model_mean')
plt.legend(loc='best', fancybox=True)
#plt.savefig(path_name + path_file + "/distance")

plt.figure()
plt.title('Alfa and policy advantage')
plt.xlabel('Iteration')
plt.plot(iterations, p_advantages, color='b', label='policy_advantage')
plt.plot(iterations, alfas, color='b',  linestyle='dashed', label='alfa')
plt.legend(loc='best', fancybox=True)
plt.yscale('log')
#plt.savefig(path_name + path_file + "/alfa_advantage")



#print('The script took {0} minutes'.format((time.time() - startTime) / 60))
