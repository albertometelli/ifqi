from ifqi.envs.lqg1d import LQG1D
from ifqi.algorithms.policy_gradient.onoff_learner import OnOffLearner
from ifqi.algorithms.policy_gradient.policy import *
from ifqi.evaluation.trajectory_generator import OnlineTrajectoryGenerator, \
    OfflineTrajectoryGenerator
from ifqi.algorithms.policy_gradient.policy_gradient_learner import PolicyGradientLearner
import matplotlib.pyplot as plt
import numpy as np

mdp = LQG1D()
mdp.horizon = 20
learn_sigma = True
initial_sigma = 1.
initial_mu = -0.2

'''
if learn_sigma:
    behavioral_policy = GaussianPolicyLinearMeanCholeskyVar(initial_mu,
                                                            initial_sigma)
    target_policy = GaussianPolicyLinearMeanCholeskyVar(initial_mu,
                                                        initial_sigma)
else:
    behavioral_policy = GaussianPolicyLinearMean(initial_mu,
                                                 initial_sigma)
    target_policy = GaussianPolicyLinearMean(initial_mu,
                                             initial_sigma)
'''

def feature(state):
    return np.array([state, 1.]).ravel()

initial_parameter = [0., 0.]
initial_sigma = 1.

target_policy = GaussianPolicyLinearFeatureMeanCholeskyVar(feature, initial_parameter, initial_sigma, max_feature=1.)
behavioral_policy = GaussianPolicyLinearFeatureMeanCholeskyVar(feature, initial_parameter, initial_sigma, max_feature=1.)



learner = OnOffLearner(mdp,
                       behavioral_policy,
                       target_policy,
                       initial_batch_size=200,
                       batch_size_incr=10,
                       max_batch_size=3000,
                       select_initial_point=False,
                       select_optimal_horizon=False,
                       adaptive_stop=True,
                       optimize_bound=True,
                       safe_stopping=True,
                       search_horizon=True,
                       adapt_batchsize=False,
                       search_step_size=True,
                       bound='student',
                       delta=0.2,
                       importance_weighting_method='pdis',
                       learning_rate=10.,
                       estimator='gpomdp',
                       gradient_updater='adanat',
                       gradient_updater_outer='vanilla',
                       max_offline_iterations=100,
                       online_iterations=100,
                       verbose=2,
                       file_offline_epochs='lqg_offline_stud.csv',
                       file_online_epochs='lqg_online_stud.csv',
                       natural=False,
                       normalize_return=False
                    )

optimal_parameter, history, history_filter = learner.learn()
history_filter = np.unique(history_filter)
np.save('./history',history)
np.save('./history_filter',history_filter)

plt.plot(np.array(history)[history_filter])

'''

target_policy = GaussianPolicyLinearMeanCholeskyVar(mu, sigma)

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
                                               gradient_updater='annelling',
                                               max_iter_opt=100,
                                               max_iter_eval=200,
                                               verbose=2)

initial_parameter = target_policy.get_parameter()
optimal_parameter, history_online_reinforce = online_reinforce_cheb.optimize(
        initial_parameter, return_history=True)

fig, ax = plt.subplots()
ax.plot(np.vstack(np.array(history)[:, 0])[:, 0], 'r', label='Ours')
ax.scatter(history_filter, np.vstack(np.array(history)[:, 0])[history_filter, 0], c='r', marker='o')
ax.set_xlabel('Iteration')
ax.set_ylabel('Parameter')
legend = ax.legend(loc='upper right')

fig, ax = plt.subplots()
ax.plot(np.vstack(np.array(history)[:, 0])[:,1], 'b', label='Ours')
ax.scatter(history_filter, np.vstack(np.array(history)[:, 0])[history_filter, 1], c='b', marker='o')
ax.set_xlabel('Iteration')
ax.set_ylabel('Std')
legend = ax.legend(loc='upper right')

fig, ax = plt.subplots()
ax.plot(np.array(history)[:, 1], 'g', label='Ours')
ax.scatter(history_filter, np.vstack(np.array(history)[:, 1])[history_filter], c='g', marker='o')
ax.set_xlabel('Iteration')
ax.set_ylabel('Avg return')
legend = ax.legend(loc='upper right')



fig, ax = plt.subplots()
ax.plot(np.vstack(np.array(history)[:, 0])[history_filter, 0], 'r', label='Ours')
ax.plot(np.vstack(np.array(history_online_reinforce)[:, 0])[:, 0], 'r:', label='Online GPOMDP')
ax.set_xlabel('Iteration')
ax.set_ylabel('Parameter')
legend = ax.legend(loc='upper right')

fig, ax = plt.subplots()
ax.plot(np.vstack(np.array(history)[:, 0])[history_filter, 1], 'b', label='Ours')
ax.plot(np.vstack(np.array(history_online_reinforce)[:, 0])[:, 1], 'b--', label='Online GPOMDP')
ax.set_xlabel('Iteration')
ax.set_ylabel('Std')
legend = ax.legend(loc='upper right')

fig, ax = plt.subplots()
ax.plot(np.array(history)[history_filter[:-1], 1], 'g', label='Ours')
ax.plot(np.array(history_online_reinforce)[:-1, 1], 'g:', label='Online GPOMDP')
ax.set_xlabel('Iteration')
ax.set_ylabel('Avg return')
legend = ax.legend(loc='upper right')

def lqg_theoreical_J(k, sigma, lambda_, r, q, a, b, S, gamma):
    return -1./(1-gamma) * sigma**2 * r - (k**2*r+q)/(1-gamma*(a+k*b)**2) * (S**2/3 + gamma/(1-gamma) * (lambda_**2+b**2*sigma**2))

J_theoretical_ours = []
for i in range(len(np.array(history)[history_filter])):
    J_theoretical_ours.append(np.asscalar(lqg_theoreical_J(np.array(history)[history_filter][i][0][0], np.array(history)[history_filter][i][0][1], mdp.sigma_noise, mdp.R, mdp.Q, mdp.A, mdp.B, mdp.max_pos, mdp.gamma)))

J_theoretical_gpomdp = []
for i in range(len(history_online_reinforce)):
    J_theoretical_gpomdp.append(np.asscalar(lqg_theoreical_J(history_online_reinforce[i][0][0], history_online_reinforce[i][0][1], mdp.sigma_noise, mdp.R, mdp.Q, mdp.A, mdp.B, mdp.max_pos, mdp.gamma)))


fig, ax = plt.subplots()
ax.plot(J_theoretical_ours, 'g', label='Ours')
ax.plot(J_theoretical_gpomdp, 'g:', label='GPOMDP')
ax.set_xlabel('Iteration')
ax.set_ylabel('Expected return theoretical')
legend = ax.legend(loc='upper right')
'''