from ifqi.envs.lqg1d import LQG1D
from ifqi.algorithms.policy_gradient.policy import GaussianPolicyLinearMean, DeterministicPolicyLinearMean
from ifqi.evaluation.evaluation import collect_episodes
from ifqi.evaluation.trajectory_generator import OnlineTrajectoryGenerator, OfflineTrajectoryGenerator
from ifqi.algorithms.policy_gradient.policy_gradient_learner import PolicyGradientLearner
import numpy as np
import matplotlib.pyplot as plt

mdp = LQG1D()
K_opt = mdp.computeOptimalK()

sb = 2
st = 1
mub = -0.2
mut = -0.2
N = 5000

#Instantiate policies
optimal_policy = DeterministicPolicyLinearMean(K_opt)
behavioral_policy = GaussianPolicyLinearMean(mub, sb**2)
target_policy = GaussianPolicyLinearMean(mut, st**2)

#Evaluate optimal policy
J_opt = []
for i in range(100):
    dataset = collect_episodes(mdp, optimal_policy, n_episodes=1)
    J_opt.append(np.dot(dataset[:,2], mdp.gamma ** np.arange(len(dataset))))
J_opt = np.mean(J_opt)

#Collect trajectories
dataset = collect_episodes(mdp, behavioral_policy, n_episodes=N)

offline_trajectory_generator = OfflineTrajectoryGenerator(dataset)
online_trajectory_generator = OnlineTrajectoryGenerator(mdp, target_policy)
'''
online_reinforce = PolicyGradientLearner(online_trajectory_generator,
                                  target_policy,
                                  mdp.gamma,
                                  mdp.horizon,
                                  learning_rate=0.002,
                                  estimator='reinforce',
                                  gradient_updater='adam',
                                  max_iter_opt=200,
                                  max_iter_eval=100,
                                  verbose=1)

online_gpomdp = PolicyGradientLearner(online_trajectory_generator,
                                  target_policy,
                                  mdp.gamma,
                                  mdp.horizon,
                                  learning_rate=0.002,
                                  estimator='gpomdp',
                                  gradient_updater='adam',
                                  max_iter_opt=200,
                                  max_iter_eval=100,
                                  verbose=1)

offline_reinforce = PolicyGradientLearner(online_trajectory_generator,
                                  target_policy,
                                  mdp.gamma,
                                  mdp.horizon,
                                  behavioral_policy=behavioral_policy,
                                  importance_weighting_method='pdis',
                                  learning_rate=0.002,
                                  estimator='reinforce',
                                  gradient_updater='adam',
                                  max_iter_opt=200,
                                  max_iter_eval=100,
                                  verbose=1)

offline_gpomdp = PolicyGradientLearner(online_trajectory_generator,
                                  target_policy,
                                  mdp.gamma,
                                  mdp.horizon,
                                  behavioral_policy=behavioral_policy,
                                  importance_weighting_method='pdis',
                                  learning_rate=0.002,
                                  estimator='gpomdp',
                                  gradient_updater='adam',
                                  max_iter_opt=200,
                                  max_iter_eval=100,
                                  verbose=1)
'''
is_offline_reinforce = PolicyGradientLearner(online_trajectory_generator,
                                  target_policy,
                                  mdp.gamma,
                                  mdp.horizon,
                                  behavioral_policy=behavioral_policy,
                                  importance_weighting_method='is',
                                  learning_rate=0.002,
                                  estimator='reinforce',
                                  gradient_updater='adam',
                                  max_iter_opt=200,
                                  max_iter_eval=100,
                                  verbose=1)

is_offline_gpomdp = PolicyGradientLearner(online_trajectory_generator,
                                  target_policy,
                                  mdp.gamma,
                                  mdp.horizon,
                                  behavioral_policy=behavioral_policy,
                                  importance_weighting_method='is',
                                  learning_rate=0.002,
                                  estimator='gpomdp',
                                  gradient_updater='adam',
                                  max_iter_opt=200,
                                  max_iter_eval=100,
                                  verbose=1)

initial_parameter = target_policy.from_param_to_vec(-0.2)
#_, history_online_reinforce = online_reinforce.optimize(initial_parameter, return_history=True)
#_, history_online_gpomdp = online_gpomdp.optimize(initial_parameter, return_history=True)
#_, history_offline_reinforce = offline_reinforce.optimize(initial_parameter, return_history=True)
#_, history_offline_gpomdp = offline_gpomdp.optimize(initial_parameter, return_history=True)
_, history_is_offline_reinforce = is_offline_reinforce.optimize(initial_parameter, return_history=True)
_, history_is_offline_gpomdp = is_offline_gpomdp.optimize(initial_parameter, return_history=True)

fig, ax = plt.subplots()
#ax.plot(np.array(history_reinforce)[:, 1], 'r', label='Reinforce')
#ax.plot(np.array(history_online_reinforce)[:, 0], 'r', label='Online REINFORCE')
#ax.plot(np.array(history_online_gpomdp)[:, 0], 'b', label='Online GPOMDP')
#ax.plot(np.array(history_offline_reinforce)[:, 0], 'r--', label='Offline PDIS REINFORCE')
#ax.plot(np.array(history_offline_gpomdp)[:, 0], 'b--', label='Offline PDIS GPOMDP')
ax.plot(np.array(history_is_offline_reinforce)[:, 0], 'r:', label='Offline IS REINFORCE')
ax.plot(np.array(history_is_offline_gpomdp)[:, 0], 'b:', label='Offline IS GPOMDP')
#ax.plot([0, 199], [J_opt, J_opt], 'k', label='Optimal')
legend = ax.legend(loc='lower right')
