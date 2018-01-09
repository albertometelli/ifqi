from ifqi.envs.lqg1d import LQG1D
from ifqi.algorithms.policy_gradient.policy import GaussianPolicyLinearMean, DeterministicPolicyLinearMean
from ifqi.evaluation.evaluation import collect_episodes
from ifqi.evaluation.trajectory_generator import OnlineTrajectoryGenerator, OfflineTrajectoryGenerator
from ifqi.algorithms.policy_gradient.policy_gradient_learner import PolicyGradientLearner
import numpy as np
import matplotlib.pyplot as plt

mdp = LQG1D()
K_opt = mdp.computeOptimalK()

sb = 2.
st = 1.
mub = -0.2
mut = 0.
N = 5000

iterations = 200
batch_size = 1000

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

offline_reinforce_cheb = PolicyGradientLearner(offline_trajectory_generator,
                                  target_policy,
                                  mdp.gamma,
                                  mdp.horizon,
                                  select_initial_point=True,
                                  bound='chebyshev',
                                  delta=0.2,
                                  behavioral_policy=behavioral_policy,
                                  importance_weighting_method='is',
                                  learning_rate=0.002,
                                  estimator='gpomdp',
                                  gradient_updater='adam',
                                  max_iter_opt=iterations,
                                  max_iter_eval=batch_size,
                                  verbose=1)

offline_reinforce_hoeff = PolicyGradientLearner(offline_trajectory_generator,
                                  target_policy,
                                  mdp.gamma,
                                  mdp.horizon,
                                  select_initial_point=True,
                                  bound='hoeffding',
                                  delta=0.2,
                                  behavioral_policy=behavioral_policy,
                                  importance_weighting_method='is',
                                  learning_rate=0.002,
                                  estimator='gpomdp',
                                  gradient_updater='adam',
                                  max_iter_opt=iterations,
                                  max_iter_eval=batch_size,
                                  verbose=1)

offline_reinforce_bern = PolicyGradientLearner(offline_trajectory_generator,
                                  target_policy,
                                  mdp.gamma,
                                  mdp.horizon,
                                  select_initial_point=True,
                                  bound='bernstein',
                                  delta=0.2,
                                  behavioral_policy=behavioral_policy,
                                  importance_weighting_method='is',
                                  learning_rate=0.002,
                                  estimator='gpomdp',
                                  gradient_updater='adam',
                                  max_iter_opt=iterations,
                                  max_iter_eval=batch_size,
                                  verbose=1)

offline_reinforce = PolicyGradientLearner(offline_trajectory_generator,
                                  target_policy,
                                  mdp.gamma,
                                  mdp.horizon,
                                  select_initial_point=False,
                                  behavioral_policy=behavioral_policy,
                                  importance_weighting_method='is',
                                  learning_rate=0.002,
                                  estimator='gpomdp',
                                  gradient_updater='adam',
                                  max_iter_opt=iterations,
                                  max_iter_eval=batch_size,
                                  verbose=1)

initial_parameter = target_policy.from_param_to_vec(mut)

_, history_offline_reinforce_hoeff = offline_reinforce_hoeff.optimize(initial_parameter, return_history=True)
_, history_offline_reinforce_bern = offline_reinforce_bern.optimize(initial_parameter, return_history=True)
_, history_offline_reinforce_cheb = offline_reinforce_cheb.optimize(initial_parameter, return_history=True)
_, history_offline_reinforce = offline_reinforce.optimize(initial_parameter, return_history=True)


fig, ax = plt.subplots()
ax.plot(np.array(history_offline_reinforce_hoeff)[:, 0], 'r', label='Offline Hoeffding')
ax.plot(np.array(history_offline_reinforce_cheb)[:, 0], 'g', label='Offline Chebyshev')
ax.plot(np.array(history_offline_reinforce_bern)[:, 0], 'y', label='Offline Bernstein')
ax.plot(np.array(history_offline_reinforce)[:, 0], 'b', label='Offline no bound')
ax.plot([0, iterations], [np.asscalar(K_opt)]*2 ,'k', label='Optimal')
ax.set_xlabel('Iteration')
ax.set_ylabel('Parameter')
legend = ax.legend(loc='upper right')

fig, ax = plt.subplots()
ax.plot(np.array(history_offline_reinforce_hoeff)[:, 1], 'r', label='Offline Hoeffding')
ax.plot(np.array(history_offline_reinforce_cheb)[:, 1], 'g', label='Offline Chebyshev')
ax.plot(np.array(history_offline_reinforce_bern)[:, 1], 'y', label='Offline Bernstein')
ax.plot(np.array(history_offline_reinforce)[:, 1], 'b', label='Offline no bound')
ax.plot([0, iterations], [J_opt]*2 ,'k', label='Optimal')
ax.set_xlabel('Iteration')
ax.set_ylabel('J(K:H)')
legend = ax.legend(loc='lower right')

fig, ax = plt.subplots()
ax.plot(np.array(history_offline_reinforce_hoeff)[:, 1] + np.array(history_offline_reinforce_hoeff)[:, 3] ,  'r', label='Offline Hoeffding')
ax.plot(np.array(history_offline_reinforce_cheb)[:, 1] + np.array(history_offline_reinforce_cheb)[:, 3], 'g', label='Offline Chebyshev')
ax.plot(np.array(history_offline_reinforce_bern)[:, 1] + np.array(history_offline_reinforce_bern)[:, 3], 'y', label='Offline Bernstein')
ax.plot(np.array(history_offline_reinforce)[:, 1] + np.array(history_offline_reinforce)[:, 3], 'b', label='Offline no bound')
ax.plot([0, iterations], [J_opt]*2 ,'k', label='Optimal')
ax.set_xlabel('Iteration')
ax.set_ylabel('Bound')
legend = ax.legend(loc='lower right')

fig, ax = plt.subplots()
ax.plot(np.array(history_offline_reinforce_hoeff)[:, 4], 'r', label='Offline Hoeffding')
ax.plot(np.array(history_offline_reinforce_cheb)[:, 4], 'g', label='Offline Chebyshev')
ax.plot(np.array(history_offline_reinforce_bern)[:, 4], 'y', label='Offline Bernstein')
ax.plot(np.array(history_offline_reinforce)[:, 4], 'b', label='Offline no bound')
ax.set_xlabel('Iteration')
ax.set_ylabel('H*')
legend = ax.legend(loc='lower right')

