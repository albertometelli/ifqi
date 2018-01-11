from ifqi.envs.lqg1d import LQG1D
from ifqi.algorithms.policy_gradient.policy import GaussianPolicyLinearMean, \
    DeterministicPolicyLinearMean
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

sb = 2.
st = 1.
mub = -0.2
mut = 0.

# Instantiate policies
optimal_policy = DeterministicPolicyLinearMean(K_opt)
behavioral_policy = GaussianPolicyLinearMean(mub, sb ** 2)
target_policy = GaussianPolicyLinearMean(mut, st ** 2)


N = 500

online_iterations = 10
offline_iterations = 10

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
                                                   verbose=0)

    initial_parameter = behavioral_policy.get_parameter()
    print(initial_parameter)
    optimal_parameter, history_offline_reinforce = offline_reinforce_cheb.optimize(
        initial_parameter, return_history=True)

    history.append(history_offline_reinforce)

    behavioral_policy.set_parameter(optimal_parameter)
    target_policy.set_parameter(optimal_parameter)

fig, ax = plt.subplots()
ax.plot(np.array(np.vstack(history))[:, 0], 'r', label='Offline')
ax.set_xlabel('Iteration')
ax.set_ylabel('Parameter')
legend = ax.legend(loc='upper right')
