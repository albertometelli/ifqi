from ifqi.envs.lqg1d import LQG1D
from ifqi.algorithms.policy_gradient.policy import GaussianPolicyLinearMean, \
    DeterministicPolicyLinearMean, GaussianPolicyLinearMeanCholeskyVar
from ifqi.evaluation.evaluation import collect_episodes
from ifqi.evaluation.trajectory_generator import OnlineTrajectoryGenerator, \
    OfflineTrajectoryGenerator
from ifqi.algorithms.policy_gradient.policy_gradient_learner import \
    PolicyGradientLearner
import numpy as np
import matplotlib.pyplot as plt

mdp = LQG1D()
mdp.horizon = 15
K_opt = mdp.computeOptimalK()

sb = 1.
st = 1.
mub = -0.2
mut = -0.2

# Instantiate policies
optimal_policy = DeterministicPolicyLinearMean(K_opt)
behavioral_policy = GaussianPolicyLinearMeanCholeskyVar(mub, sb)
target_policy = GaussianPolicyLinearMeanCholeskyVar(mut, st)


N = 1000

online_iterations = 10
offline_iterations = 20

history = []

for i in range(online_iterations):
    dataset = collect_episodes(mdp, behavioral_policy, n_episodes=N)
    offline_trajectory_generator = OfflineTrajectoryGenerator(dataset)

    offline_reinforce_cheb = PolicyGradientLearner(offline_trajectory_generator,
                                                   target_policy,
                                                   mdp.gamma,
                                                   mdp.horizon,
                                                   select_initial_point=False,
                                                   select_optimal_horizon=False,
                                                   bound='chebyshev',
                                                   delta=0.2,
                                                   behavioral_policy=behavioral_policy,
                                                   importance_weighting_method='is',
                                                   learning_rate=0.002,
                                                   estimator='gpomdp',
                                                   gradient_updater='adam',
                                                   max_iter_opt=offline_iterations,
                                                   max_iter_eval=500,
                                                   verbose=2)

    initial_parameter = behavioral_policy.get_parameter()
    optimal_parameter, history_offline_reinforce = offline_reinforce_cheb.optimize(
        initial_parameter, return_history=True)

    history.extend(history_offline_reinforce[:-1])
    behavioral_policy.set_parameter(optimal_parameter)
    target_policy.set_parameter(optimal_parameter)

fig, ax = plt.subplots()
ax.plot(np.vstack(np.array(history)[:, 0])[:,0], 'r', label='Offline')
ax.scatter(range(0,offline_iterations*online_iterations,online_iterations), np.vstack(np.array(history)[:, 0])[:,0][range(0,offline_iterations*online_iterations,online_iterations)], marker='o')
ax.set_xlabel('Iteration')
ax.set_ylabel('Parameter')
legend = ax.legend(loc='upper right')

fig, ax = plt.subplots()
ax.plot(np.vstack(np.array(history)[:, 0])[:,1], 'b', label='Offline')
ax.scatter(range(0,offline_iterations*online_iterations,online_iterations), np.vstack(np.array(history)[:, 0])[:,1][range(0,offline_iterations*online_iterations,online_iterations)], marker='o')
ax.set_xlabel('Iteration')
ax.set_ylabel('Std')
legend = ax.legend(loc='upper right')

fig, ax = plt.subplots()
ax.plot(np.array(history)[:, 1], 'g', label='Offline')
ax.scatter(range(0,offline_iterations*online_iterations,online_iterations), np.array(history)[:, 1][range(0,offline_iterations*online_iterations,online_iterations)], marker='o')
ax.set_xlabel('Iteration')
ax.set_ylabel('Avg return')
legend = ax.legend(loc='upper right')

