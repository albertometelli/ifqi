import numpy as np
from ifqi.envs.continuous_mountain_car import Continuous_MountainCarEnv
from ifqi.algorithms.policy_gradient.policy import RBFGaussianPolicy, GaussianPolicyLinearMeanFeatures
from ifqi.evaluation.evaluation import collect_episodes
from ifqi.evaluation.trajectory_generator import OfflineTrajectoryGenerator, OnlineTrajectoryGenerator
from ifqi.algorithms.policy_gradient.policy_gradient_learner import PolicyGradientLearner
import matplotlib.pyplot as plt

mdp = Continuous_MountainCarEnv()
N = 1000
iterations = 50
batch_size = 300

def features(state):
    position, speed = state[0], state[1]
    return np.array([position > 0 and speed > 0,
                     position < 0 and speed > 0,
                     position > 0 and speed < 0,
                     position < 0 and speed < 0], dtype=int)

behavioral_policy_parameters = [0.6, 0.4, -0.5, -0.3]
behavioral_policy = GaussianPolicyLinearMeanFeatures(features, behavioral_policy_parameters, 1.)

target_policy_parameters = [0., 0., 0., 0.]
target_policy = GaussianPolicyLinearMeanFeatures(features, target_policy_parameters, 0.8)
'''
for _ in range(5):
    episode = collect_episodes(mdp, behavioral_policy, n_episodes=1)
    print(len(episode))
    for i in range(len(episode)):
        s = episode[i, [0,1]]
        mdp.state = s
        mdp._render()
    print(sum(episode[:, 3]))
    print(np.dot(episode[:, 3], 0.99**np.arange(len(episode))))
'''

'''
#Build a policy
n_dim_centers = 50
n_centers = n_dim_centers * n_dim_centers
centers = np.meshgrid(np.linspace(mdp.min_position, mdp.max_position, n_dim_centers),
                      np.linspace(-mdp.max_speed, mdp.max_speed, n_dim_centers))

centers = np.vstack([centers[0].ravel(), centers[1].ravel()]).T

pp = np.bitwise_and(centers[:,0]>=0, centers[:,1]>0)
pn = np.bitwise_and(centers[:,0]>=0, centers[:,1]<0)
np_ = np.bitwise_and(centers[:,0]<0, centers[:,1]>0)
nn = np.bitwise_and(centers[:,0]<=0, centers[:,1]<=0)

class IfPolicy(object):

    def draw_action(self, state, done):
        if state[0] >= 0 and state[1] > 0 or state[0] < 0 and state[1] > 0:
            return 1.
        return -1.

parameters = np.zeros(n_centers)

parameters[pp] = +.3
parameters[pn] = -.3
parameters[np_] = +.3
parameters[nn] = -.3

behavioral_policy = RBFGaussianPolicy(centers, parameters, sigma=0.1, radial_basis_parameters=1.)
target_policy = RBFGaussianPolicy(centers, parameters, sigma=0.1, radial_basis_parameters=1.)

ifpolicy = IfPolicy()

for _ in range(5):
    episode = collect_episodes(mdp, ifpolicy, n_episodes=1)
    print(len(episode))
    for i in range(len(episode)):
        s = episode[i, [0,1]]
        mdp.state = s
        mdp._render()
    print(sum(episode[:, 3]))
    print(np.dot(episode[:, 3], 0.99**np.arange(len(episode))))

'''
#Collect trajectories
dataset = collect_episodes(mdp, behavioral_policy, n_episodes=N)
offline_trajectory_generator = OfflineTrajectoryGenerator(dataset)
'''
online_trajectory_generator = OnlineTrajectoryGenerator(mdp, behavioral_policy)
online_gpomdp = PolicyGradientLearner(online_trajectory_generator,
                                  behavioral_policy,
                                  mdp.gamma,
                                  mdp.horizon,
                                  learning_rate=0.01,
                                  estimator='gpomdp',
                                  gradient_updater='adam',
                                  max_iter_opt=iterations,
                                  max_iter_eval=50,
                                  verbose=1,
                                  state_index=[0,1],
                                  action_index=2,
                                  reward_index=3)

theta, history = online_gpomdp.optimize(behavioral_policy_parameters, return_history=True)

plt.plot(np.array(history)[:, 1])
'''

offline_gpomdp = PolicyGradientLearner(offline_trajectory_generator,
                                  target_policy,
                                  mdp.gamma,
                                  mdp.horizon,
                                  select_initial_point=False,
                                  behavioral_policy=behavioral_policy,
                                  importance_weighting_method='is',
                                  learning_rate=0.01,
                                  estimator='gpomdp',
                                  gradient_updater='adam',
                                  max_iter_opt=iterations,
                                  max_iter_eval=batch_size,
                                  verbose=1,
                                  state_index=[0,1],
                                  action_index=2,
                                  reward_index=3)

_, history_offline = offline_gpomdp.optimize(behavioral_policy_parameters, return_history=True)


offline_gpomdp_cheb = PolicyGradientLearner(offline_trajectory_generator,
                                  target_policy,
                                  mdp.gamma,
                                  mdp.horizon,
                                  bound='chebyshev',
                                  delta=0.2,
                                  select_initial_point=True,
                                  behavioral_policy=behavioral_policy,
                                  importance_weighting_method='is',
                                  learning_rate=0.01,
                                  estimator='gpomdp',
                                  gradient_updater='adam',
                                  max_iter_opt=iterations,
                                  max_iter_eval=batch_size,
                                  verbose=1,
                                  state_index=[0,1],
                                  action_index=2,
                                  reward_index=3)

_, history_offline_cheb = offline_gpomdp_cheb.optimize(behavioral_policy_parameters, return_history=True)

offline_gpomdp_hoeff = PolicyGradientLearner(offline_trajectory_generator,
                                  target_policy,
                                  mdp.gamma,
                                  mdp.horizon,
                                  bound='hoeffding',
                                  delta=0.2,
                                  select_initial_point=True,
                                  behavioral_policy=behavioral_policy,
                                  importance_weighting_method='is',
                                  learning_rate=0.01,
                                  estimator='gpomdp',
                                  gradient_updater='adam',
                                  max_iter_opt=iterations,
                                  max_iter_eval=batch_size,
                                  verbose=1,
                                  state_index=[0,1],
                                  action_index=2,
                                  reward_index=3)

_, history_offline_hoeff = offline_gpomdp_hoeff.optimize(behavioral_policy_parameters, return_history=True)

offline_gpomdp_bern = PolicyGradientLearner(offline_trajectory_generator,
                                  target_policy,
                                  mdp.gamma,
                                  mdp.horizon,
                                  bound='bernstein',
                                  delta=0.2,
                                  select_initial_point=True,
                                  behavioral_policy=behavioral_policy,
                                  importance_weighting_method='is',
                                  learning_rate=0.01,
                                  estimator='gpomdp',
                                  gradient_updater='adam',
                                  max_iter_opt=iterations,
                                  max_iter_eval=batch_size,
                                  verbose=1,
                                  state_index=[0,1],
                                  action_index=2,
                                  reward_index=3)

_, history_offline_bern = offline_gpomdp_bern.optimize(behavioral_policy_parameters, return_history=True)
