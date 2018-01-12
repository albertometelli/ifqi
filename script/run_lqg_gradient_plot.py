from ifqi.envs.lqg1d import LQG1D
from ifqi.algorithms.policy_gradient.policy import GaussianPolicyLinearMean, \
    DeterministicPolicyLinearMean, GaussianPolicyLinearMeanCholeskyVar
from ifqi.evaluation.evaluation import collect_episodes
from ifqi.evaluation.trajectory_generator import OnlineTrajectoryGenerator, \
    OfflineTrajectoryGenerator
from ifqi.algorithms.policy_gradient.policy_gradient_learner import \
    PolicyGradientLearner
from ifqi.algorithms.bound import bound_factory
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



N = 100

online_iterations = 100
offline_iterations = 50

history = []
lens = [0]

k = np.arange(-0.7, 0.1, 0.1)
lam = np.arange(0.1, 1.4, 0.1)

K, L = np.meshgrid(k, lam)

gradient_penalization = np.zeros((K.shape[0], K.shape[1], 2))
gradient = np.zeros((K.shape[0], K.shape[1], 2))

dataset = collect_episodes(mdp, behavioral_policy, n_episodes=N)
offline_trajectory_generator = OfflineTrajectoryGenerator(dataset)

for i in range(K.shape[0]):
    for j in range(K.shape[1]):

        target_policy.set_parameter(target_policy.from_param_to_vec(K[i,j], L[i, j]))

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
                                                       max_iter_opt=1,
                                                       max_iter_eval=N,
                                                       verbose=2)

        #optimal_parameter, history_offline_reinforce = offline_reinforce_cheb.optimize(
        #    target_policy.from_param_to_vec(K[i, j], L[i, j]), return_history=True)

        gradient_penalization[i,j] = offline_reinforce_cheb.bound.gradient_penalization()
        #gradient[i,j] = history_offline_reinforce[0][3]

print(gradient_penalization.shape)
rho = np.apply_along_axis(lambda x: np.linalg.norm(x), 2, gradient_penalization)
rho = np.log(rho + 1)
theta = np.apply_along_axis(lambda x: np.arctan2(x[1], x[0]), 2, gradient_penalization)
gradient_penalization = np.array([rho * np.cos(theta),  rho * np.sin(theta)]).transpose((1,2,0))
print(gradient_penalization.shape)

_max = gradient_penalization.max()
gradient_penalization = gradient_penalization / _max * 0.075

fig, ax = plt.subplots()
for i in range(K.shape[0]):
    for j in range(K.shape[1]):
        #r = np.sqrt(gradient_penalization[i, j, 0]**2 + gradient_penalization[i, j, 1]**2)/0.05
        ax.arrow(K[i, j], L[i, j], gradient_penalization[i, j, 0],
                 gradient_penalization[i, j, 1], head_width=0.01,
                 head_length=0.02, fc='k', ec='k')

ax.set_xlim([-0.7, 0])
ax.set_ylim([0, 1.5])

plt.show()