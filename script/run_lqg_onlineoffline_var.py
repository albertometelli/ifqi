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
mdp.horizon = 5
K_opt = mdp.computeOptimalK()

sb = 1.
st = 1.
mub = -0.2
mut = -0.2

# Instantiate policies
optimal_policy = DeterministicPolicyLinearMean(K_opt)
behavioral_policy = GaussianPolicyLinearMeanCholeskyVar(mub, sb)
target_policy = GaussianPolicyLinearMeanCholeskyVar(mut, st)


N = 200

online_iterations = 10
offline_iterations = 20

history = []
lens = [0]

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
                                                   learning_rate=0.001,
                                                   estimator='gpomdp',
                                                   gradient_updater='vanilla',
                                                   max_iter_opt=offline_iterations,
                                                   max_iter_eval=N,
                                                   verbose=2)

    initial_parameter = behavioral_policy.get_parameter()
    optimal_parameter, history_offline_reinforce = offline_reinforce_cheb.optimize(
        initial_parameter, return_history=True)

    lens.append(len(history_offline_reinforce)-1)
    history.extend(history_offline_reinforce[:-1])
    behavioral_policy.set_parameter(optimal_parameter)
    target_policy.set_parameter(optimal_parameter)

history.append(history_offline_reinforce[-1])
_filter = np.cumsum(lens)

target_policy.set_parameter(target_policy.from_param_to_vec(-0.2, 1.))
online_trajectory_generator = OnlineTrajectoryGenerator(mdp, target_policy)

online_reinforce_cheb = PolicyGradientLearner(online_trajectory_generator,
                                               target_policy,
                                               mdp.gamma,
                                               mdp.horizon,
                                               select_initial_point=False,
                                               select_optimal_horizon=False,
                                               learning_rate=0.001,
                                               estimator='gpomdp',
                                               gradient_updater='vanilla',
                                               max_iter_opt=sum(lens),
                                               max_iter_eval=N,
                                               verbose=2)

initial_parameter = target_policy.get_parameter()
optimal_parameter, history_online_reinforce = online_reinforce_cheb.optimize(
        initial_parameter, return_history=True)

fig, ax = plt.subplots()
ax.plot(np.vstack(np.array(history)[:, 0])[:, 0], 'r', label='Offline')
ax.plot(np.vstack(np.array(history_online_reinforce)[:, 0])[:, 0], 'r--', label='Online++')
ax.plot(_filter, np.vstack(np.array(history_online_reinforce)[:, 0])[:len(_filter), 0], 'r:', label='Online--')
ax.scatter(_filter, np.vstack(np.array(history)[:, 0])[_filter, 0], c='r', marker='o')
ax.set_xlabel('Iteration')
ax.set_ylabel('Parameter')
legend = ax.legend(loc='upper right')

fig, ax = plt.subplots()
ax.plot(np.vstack(np.array(history)[:, 0])[:,1], 'b', label='Offline')
ax.plot(np.vstack(np.array(history_online_reinforce)[:, 0])[:, 1], 'b--', label='Online++')
ax.plot(_filter, np.vstack(np.array(history_online_reinforce)[:len(_filter), 0])[:, 1], 'b:', label='Online--')
ax.scatter(_filter, np.vstack(np.array(history)[:, 0])[_filter, 1], c='b', marker='o')
ax.set_xlabel('Iteration')
ax.set_ylabel('Std')
legend = ax.legend(loc='upper right')

fig, ax = plt.subplots()
ax.plot(np.array(history)[:, 1], 'g', label='Offline')
ax.plot(np.array(history_online_reinforce)[:, 1], 'g--', label='Offline++')
ax.plot(_filter, np.array(history_online_reinforce)[:len(_filter), 1], 'g:', label='Offline--')
ax.scatter(_filter, np.vstack(np.array(history)[:, 1])[_filter], c='g', marker='o')
ax.set_xlabel('Iteration')
ax.set_ylabel('Avg return')
legend = ax.legend(loc='upper right')

'''
fig, ax = plt.subplots()
ax.plot(np.array(history)[:, 1] / np.concatenate([[1], np.repeat(np.arange(len(lens)), lens)]), 'g', label='Offline')
ax.plot(np.array(history_online_reinforce)[:, 1] / np.arange(1,len(np.array(history_online_reinforce)[:, 1])+1), 'g--', label='Offline++')
ax.plot(_filter, np.array(history_online_reinforce)[:len(_filter), 1] / np.arange(1,len(_filter)+1), 'g:', label='Offline--')
ax.set_xlabel('Iteration')
ax.set_ylabel('Avg return per trajectory')
legend = ax.legend(loc='upper right')

'''