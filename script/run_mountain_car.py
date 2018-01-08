import numpy as np
from ifqi.envs.continuous_mountain_car import Continuous_MountainCarEnv
from ifqi.algorithms.policy_gradient.policy import RBFGaussianPolicy
from ifqi.evaluation.evaluation import collect_episodes
from ifqi.evaluation.trajectory_generator import OfflineTrajectoryGenerator, OnlineTrajectoryGenerator
from ifqi.algorithms.policy_gradient.policy_gradient_learner import PolicyGradientLearner

mdp = Continuous_MountainCarEnv()
N = 1000

#Build a policy
n_dim_centers = 20
n_centers = n_dim_centers * n_dim_centers
centers = np.meshgrid(np.linspace(mdp.min_position, mdp.max_position, n_dim_centers),
                      np.linspace(-mdp.max_speed, mdp.max_speed, n_dim_centers))

centers = np.vstack([centers[0].ravel(), centers[1].ravel()]).T

parameters = np.zeros((n_centers, 1))

behavioral_policy = RBFGaussianPolicy(centers, parameters, sigma=1., radial_basis_parameters=0.01)
target_policy = RBFGaussianPolicy(centers, parameters, sigma=0.1, radial_basis_parameters=0.01)

#Collect trajectories
#dataset = collect_episodes(mdp, behavioral_policy, n_episodes=N)
#offline_trajectory_generator = OfflineTrajectoryGenerator(dataset)

online_trajectory_generator = OnlineTrajectoryGenerator(mdp, behavioral_policy)
online_gpomdp = PolicyGradientLearner(online_trajectory_generator,
                                  target_policy,
                                  mdp.gamma,
                                  mdp.horizon,
                                  learning_rate=0.002,
                                  estimator='gpomdp',
                                  gradient_updater='adam',
                                  max_iter_opt=200,
                                  max_iter_eval=30,
                                  verbose=1,
                                  state_index=[0,1],
                                  action_index=2,
                                  reward_index=3)

theta = online_gpomdp.optimize(parameters.ravel())


