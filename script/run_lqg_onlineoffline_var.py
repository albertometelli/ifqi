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
mdp.horizon = 10
K_opt = mdp.computeOptimalK()

sb = 1.
st = 1.
mub = -0.2
mut = -0.2

# Instantiate policies
optimal_policy = DeterministicPolicyLinearMean(K_opt)

J = 0.
for i in range(100):
    d = collect_episodes(mdp, optimal_policy, n_episodes=1)
    J += np.dot(d[:, 2], 0.99 ** np.arange(len(d)))
J /= 100
print(J)


behavioral_policy = GaussianPolicyLinearMeanCholeskyVar(mub, sb)
target_policy = GaussianPolicyLinearMeanCholeskyVar(mut, st)


N = 200

online_iterations = 100
offline_iterations = 50

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
                                                   adaptive_stop=False,
                                                   bound_stop=True,
                                                   bound='chebyshev',
                                                   delta=0.2,
                                                   behavioral_policy=behavioral_policy,
                                                   importance_weighting_method='is',
                                                   learning_rate=0.002,
                                                   estimator='gpomdp',
                                                   gradient_updater='vanilla',
                                                   max_iter_opt=offline_iterations,
                                                   max_iter_eval=N,
                                                   verbose=0)

    initial_parameter = behavioral_policy.get_parameter()
    optimal_parameter, history_offline_reinforce = offline_reinforce_cheb.optimize(
        initial_parameter, return_history=True)

    lens.append(len(history_offline_reinforce)-1)
    history.extend(history_offline_reinforce[:-1])
    behavioral_policy.set_parameter(optimal_parameter)
    target_policy.set_parameter(optimal_parameter)

    print('*****************ITERATION %s/%s*****************' % (i, online_iterations))
    print('Initial parameter: %s' % initial_parameter)
    print('Optimal parameter: %s' % optimal_parameter)
    print('Iterations: %s' % (len(history_offline_reinforce) - 1))
    #print('M_2: %s')
    #print('M_inf %s')
    #print('Average return %s')
    #print('Gradient %s')
    #print('Gradient J_hat')
    #print('Gradient penalization')
    #print('Bound value')
    #print('Batch size')



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
                                               learning_rate=0.002,
                                               estimator='gpomdp',
                                               gradient_updater='vanilla',
                                               max_iter_opt=online_iterations,
                                               max_iter_eval=N,
                                               verbose=2)

initial_parameter = target_policy.get_parameter()
optimal_parameter, history_online_reinforce = online_reinforce_cheb.optimize(
        initial_parameter, return_history=True)

fig, ax = plt.subplots()
ax.plot(np.vstack(np.array(history)[:, 0])[:, 0], 'r', label='Ours')
ax.scatter(_filter, np.vstack(np.array(history)[:, 0])[_filter, 0], c='r', marker='o')
ax.set_xlabel('Iteration')
ax.set_ylabel('Parameter')
legend = ax.legend(loc='upper right')

fig, ax = plt.subplots()
ax.plot(np.vstack(np.array(history)[:, 0])[:,1], 'b', label='Ours')
ax.scatter(_filter, np.vstack(np.array(history)[:, 0])[_filter, 1], c='b', marker='o')
ax.set_xlabel('Iteration')
ax.set_ylabel('Std')
legend = ax.legend(loc='upper right')

fig, ax = plt.subplots()
ax.plot(np.array(history)[:, 1], 'g', label='Ours')
ax.scatter(_filter, np.vstack(np.array(history)[:, 1])[_filter], c='g', marker='o')
ax.set_xlabel('Iteration')
ax.set_ylabel('Avg return')
legend = ax.legend(loc='upper right')



fig, ax = plt.subplots()
ax.plot(np.vstack(np.array(history)[:, 0])[_filter, 0], 'r', label='Ours')
ax.plot(np.vstack(np.array(history_online_reinforce)[:, 0])[:, 0], 'r:', label='Online GPOMDP')
ax.set_xlabel('Iteration')
ax.set_ylabel('Parameter')
legend = ax.legend(loc='upper right')

fig, ax = plt.subplots()
ax.plot(np.vstack(np.array(history)[:, 0])[_filter, 1], 'b', label='Ours')
ax.plot(np.vstack(np.array(history_online_reinforce)[:, 0])[:, 1], 'b--', label='Online GPOMDP')
ax.set_xlabel('Iteration')
ax.set_ylabel('Std')
legend = ax.legend(loc='upper right')

fig, ax = plt.subplots()
ax.plot(np.array(history)[_filter[:-1], 1], 'g', label='Ours')
ax.plot(np.array(history_online_reinforce)[:-1, 1], 'g:', label='Online GPOMDP')
ax.set_xlabel('Iteration')
ax.set_ylabel('Avg return')
legend = ax.legend(loc='upper right')

def lqg_theoreical_J(k, sigma, lambda_, r, q, a, b, S, gamma):
    return -1./(1-gamma) * sigma**2 * r - (k**2*r+q)/(1-gamma*(a+k*b)**2) * (S**2/3 + gamma/(1-gamma) * (lambda_**2+b**2*sigma**2))

J_theoretical_ours = []
for i in range(len(np.array(history)[_filter])):
    J_theoretical_ours.append(np.asscalar(lqg_theoreical_J(np.array(history)[_filter][i][0][0], np.array(history)[_filter][i][0][1], mdp.sigma_noise, mdp.R, mdp.Q, mdp.A, mdp.B, mdp.max_pos, mdp.gamma)))

J_theoretical_gpomdp = []
for i in range(len(history_online_reinforce)):
    J_theoretical_gpomdp.append(np.asscalar(lqg_theoreical_J(history_online_reinforce[i][0][0], history_online_reinforce[i][0][1], mdp.sigma_noise, mdp.R, mdp.Q, mdp.A, mdp.B, mdp.max_pos, mdp.gamma)))


fig, ax = plt.subplots()
ax.plot(J_theoretical_ours, 'g', label='Ours')
ax.plot(J_theoretical_gpomdp, 'g:', label='GPOMDP')
ax.set_xlabel('Iteration')
ax.set_ylabel('Expected return theoretical')
legend = ax.legend(loc='upper right')
