from ifqi.envs.lqg1d import LQG1D
from ifqi.algorithms.policy_gradient.policy import GaussianPolicyLinearMean, DeterministicPolicyLinearMean
from ifqi.evaluation.evaluation import collect_episodes
from ifqi.evaluation.trajectory_generator import OnlineTrajectoryGenerator, OfflineTrajectoryGenerator
from ifqi.algorithms.policy_gradient.policy_gradient_learner import PolicyGradientLearner
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from ifqi.algorithms.importance_weighting.importance_weighting import RatioShrinkedImportanceWeighting
from mpl_toolkits.mplot3d import Axes3D

mdp = LQG1D()
K_opt = mdp.computeOptimalK()

sb = 1.
st = .5
mub = -0.2
mut = 0.
N = 1000

#Instantiate policies
optimal_policy = DeterministicPolicyLinearMean(K_opt)
behavioral_policy = GaussianPolicyLinearMean(mub, sb**2)
target_policy = GaussianPolicyLinearMean(mut, st**2)

#Collect trajectories
#trajectories = collect_episodes(mdp, behavioral_policy, n_episodes=N)
trajectories = np.load('../datasets/dataset_10000_1.0.npy')
dataset = []

#Dataset as list of trajectories
stop = np.where(trajectories[:, -1] == 1)[0] + 1
start = np.concatenate([[0], stop[:-1] - 1])
for i,j in zip(start, stop):
    dataset.append(trajectories[i:j])

def estimate(dataset, behavioral_policy, target_policy, gamma, h, k, c):
    is_estimator = RatioShrinkedImportanceWeighting(behavioral_policy, target_policy, shrinkage=c)
    eval = []
    for i in range(len(dataset)):
        traj = dataset[i][k:k+h]
        horizon = len(traj)
        w = is_estimator.weight(traj)
        traj_return = traj[:, 2] * gamma ** np.arange(horizon)
        eval.append(np.dot(w, traj_return))
    return np.mean(eval)

def bias(dataset, behavioral_policy, target_policy, gamma, h, k, c):
    return - (1 - gamma ** k) / (1 - gamma) - gamma ** k * gamma ** h / (
    1 - gamma) - gamma ** k * (1 - gamma ** h) / (1 - gamma) * (1 - c) * 2


def variance_Hoeffding(dataset, behavioral_policy, target_policy, gamma, h, k, c, delta):
    M_inf = target_policy.M_inf(behavioral_policy)
    N = len(dataset)
    return - gamma ** k * (1 - gamma ** h) / (1 - gamma) * (1 + c * (
        M_inf ** h - 1)) * np.sqrt(np.log(1 / delta) / (2 * N))

def variance_Chebychev(dataset, behavioral_policy, target_policy, gamma, h, k, c, delta):
    M_2 = target_policy.M_2(behavioral_policy)
    N = len(dataset)
    return - gamma ** k * (1 - gamma ** h) / (1 - gamma) * np.sqrt(
        (1 + c ** 2 * (M_2 ** h - 1)) / N * (1 / delta - 1))

def variance_Bernstein(dataset, behavioral_policy, target_policy, gamma, h, k, c, delta):
    M_inf = target_policy.M_inf(behavioral_policy)
    M_2 = target_policy.M_2(behavioral_policy)
    N = len(dataset)
    return - gamma ** k * (1 - gamma ** h) / (1 - gamma) * (
    2 * (1 + c * (M_inf ** h - 1)) * np.log(1 / delta) / (3 * N) + np.sqrt(
        2 * (1 + c ** 2 * (M_2 ** h - 1)) * np.log(1 / delta) / N))


h = 10
delta = 0.2
c = 1
delta_N = 1
N_0 = delta_N
N_max = 100

k_range = np.arange(0.,-1.,-.1)
n_range = range(N_0,N_max+1,delta_N)
max_bound = []
optimal_k = []
for j in range(len(n_range)):
    part_dataset = dataset[0:n_range[j]]
    bound_Chebychev = np.zeros(len(k_range))
    for i in range(len(k_range)):
            k = k_range[i]
            target_policy.set_parameter(k)
            J = estimate(part_dataset, behavioral_policy, target_policy, mdp.gamma, h, 0, c)
            bias_ = bias(part_dataset, behavioral_policy, target_policy, mdp.gamma, h, 0, c)

            var_cheb = variance_Chebychev(part_dataset,
                               behavioral_policy,
                               target_policy,
                               mdp.gamma, h, 0, c,
                               delta)

            print('N = %s \t Parameter = %s \t Shrinkage = %s \t Bias = %s\t J_hat = %s, Var = %s' % (n_range[j], k, c, bias_, J, var_cheb))

            bound_Chebychev[i] = J + bias_ + var_cheb
    max_bound.append(np.amax(bound_Chebychev))
    optimal_k.append(k_range[np.argmax(max_bound[j])])

fig, ax = plt.subplots()
ax.plot(n_range,max_bound)
ax.set_xlabel('n')
ax.set_ylabel('max bound')
ax.set_title('Chebyshev')
plt.show()

