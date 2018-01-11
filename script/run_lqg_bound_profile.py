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

sb = 2.
st = 1.99
mub = -0.2
mut = 0.
N = 36

K_grid = np.arange(-0.7, 0.1, 0.1)
#K_grid = np.array([-0.2])
c_grid = np.arange(0., 1.1, 0.1)
#c_grid = np.array([1.])

K, C = np.meshgrid(K_grid, c_grid)

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
trajectories = collect_episodes(mdp, behavioral_policy, n_episodes=N)
#trajectories = np.load('../datasets/dataset_1000.npy')
dataset = []

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

bound_Hoeffding = np.zeros(K.shape)
bound_Chebychev = np.zeros(K.shape)
bound_Bernstain = np.zeros(K.shape)

for i in range(K.shape[0]):
    for j in range(K.shape[1]):
        k = K[i, j]
        c = C[i, j]
        target_policy.set_parameter(k)
        J = estimate(dataset, behavioral_policy, target_policy, mdp.gamma, h, 0, c)
        bias_ = bias(dataset, behavioral_policy, target_policy, mdp.gamma, h, 0, c)



        var_hoeff = variance_Hoeffding(dataset,
                           behavioral_policy,
                           target_policy,
                           mdp.gamma, h, 0, c,
                           delta)
        var_cheb = variance_Chebychev(dataset,
                           behavioral_policy,
                           target_policy,
                           mdp.gamma, h, 0, c,
                           delta)
        var_bern = variance_Bernstein(dataset,
                           behavioral_policy,
                           target_policy,
                           mdp.gamma, h, 0, c,
                           delta)
        bound_Hoeffding[i, j] = J + bias_ + var_hoeff
        bound_Chebychev[i, j] = J + bias_ + var_cheb
        bound_Bernstain[i, j] = J + bias_ + var_bern

        print(
        'Parameter = %s \t Shrinkage = %s \t Bias = %s\t J_hat = %s \t bound = %s' % (
        k, c, bias_, J,  bound_Chebychev[i, j]))
'''
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(K, C , bound_Hoeffding, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
i, j = np.unravel_index(np.argmax(bound_Hoeffding), bound_Hoeffding.shape)
ax.scatter(K[i,j], C[i,j], bound_Hoeffding[i,j], s=200, c='k', marker='o')
ax.set_xlabel('parameter')
ax.set_ylabel('shrinkage')
ax.set_zlabel('bound')
ax.set_title('Hoeffding')
'''

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(K, C , bound_Chebychev, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
i, j = np.unravel_index(np.argmax(bound_Chebychev), bound_Chebychev.shape)
ax.scatter(K[i,j], C[i,j], bound_Chebychev[i,j], s=200, c='k', marker='o')
ax.set_xlabel('parameter')
ax.set_ylabel('shrinkage')
ax.set_zlabel('bound')
ax.set_title('Chebychev')

'''
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(K, C , bound_Bernstain, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
i, j = np.unravel_index(np.argmax(bound_Bernstain), bound_Bernstain.shape)
ax.scatter(K[i,j], C[i,j], bound_Bernstain[i,j], s=200, c='k', marker='o')
ax.set_xlabel('parameter')
ax.set_ylabel('shrinkage')
ax.set_zlabel('bound')
ax.set_title('Bernstein')




fig, ax = plt.subplots()
ax.plot(K[-1,:], bound_Hoeffding[-1,:])
i, j = np.unravel_index(np.argmax(bound_Hoeffding), bound_Hoeffding.shape)
ax.scatter(K[-1,j], bound_Hoeffding[i,j], s=200, c='k', marker='o')
ax.set_xlabel('parameter')
ax.set_ylabel('bound')
ax.set_title('Hoeffding')
ax.set_ylim([-2000,0])


fig, ax = plt.subplots()
ax.plot(K[-1,:], bound_Chebychev[-1,:])
i, j = np.unravel_index(np.argmax(bound_Chebychev), bound_Chebychev.shape)
ax.scatter(K[-1,j], bound_Chebychev[i,j], s=200, c='k', marker='o')
ax.set_xlabel('parameter')
ax.set_ylabel('bound')
ax.set_title('Chebychev')
ax.set_ylim([-2000,0])

fig, ax = plt.subplots()
ax.plot(K[-1,:], bound_Bernstain[-1,:])
i, j = np.unravel_index(np.argmax(bound_Bernstain), bound_Bernstain.shape)
ax.scatter(K[-1,j], bound_Bernstain[i,j], s=200, c='k', marker='o')
ax.set_xlabel('parameter')
ax.set_ylabel('bound')
ax.set_title('Bernstain')
ax.set_ylim([-2000,0])
'''