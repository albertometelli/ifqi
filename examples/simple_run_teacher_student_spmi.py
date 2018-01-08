from spmi.envs.teacher_student import TeacherStudentEnv
from spmi.evaluation.evaluation import collect_episodes
from spmi.algorithms.spmi_exact_non_par import SPMI
from spmi.utils.uniform_policy import UniformPolicy
from spmi.utils.tabular import TabularPolicy, TabularModel
from spmi.algorithms.policy_chooser import *
from spmi.algorithms.model_chooser import *
import numpy as np
from gym.utils import seeding
import matplotlib.pyplot as plt

mdp = TeacherStudentEnv(n_literals=2,
                        max_value=1,
                        max_update=1,
                        max_literals_in_examples=2,
                        horizon=10)

uniform_policy = UniformPolicy(mdp)

initial_model = TabularModel(mdp.P, mdp.nS, mdp.nA)
initial_policy = TabularPolicy(uniform_policy.get_rep(), mdp.nS, mdp.nA)

policy_chooser = GreedyPolicyChooser(mdp.nS, mdp.nA)
model_chooser = DoNotCreateTransitionsGreedyModelChooser(mdp.P, mdp.nS, mdp.nA)

eps = 0.0
delta = 0.1
spmi = SPMI(mdp, eps, policy_chooser, model_chooser, max_iter=50000, use_target_trick=True)
policy, model = spmi.safe_policy_model_iteration(initial_policy, initial_model)

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

#plt.switch_backend('pdf')

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
plt.plot(iterations, m_dist_sup, color='r', label='model_sup')
plt.plot(iterations, m_dist_mean, color='r', linestyle='dashed', label='model_mean')
plt.legend(loc='best', fancybox=True)
#plt.savefig(path_name + path_file + "/distance")

plt.figure()
plt.title('Advantage')
plt.xlabel('Iteration')
plt.plot(iterations, p_advantages, color='b', label='policy_advantage')
plt.plot(iterations, m_advantages, color='r', label='model_advantage')
plt.legend(loc='best', fancybox=True)
#plt.savefig(path_name + path_file + "/alfa_advantage")

plt.figure()
plt.title('Alfa and Beta')
plt.xlabel('Iteration')
plt.plot(iterations, alfas, color='b', marker='*', linestyle='', label='alfa')
plt.plot(iterations, betas, color='r', marker='o', linestyle='', label='beta')
plt.legend(loc='best', fancybox=True)
plt.yscale('log')

plt.figure()
plt.title('Target change')
plt.xlabel('Iteration')
plt.plot(iterations, p_change, color='b', label='policy_change')
plt.plot(iterations, m_change, color='r', label='model_change')
plt.legend(loc='best', fancybox=True)

#print('The script took {0} minutes'.format((time.time() - startTime) / 60))
