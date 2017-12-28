from ifqi.algorithms.policy_gradient.offline_learner import *
from ifqi.envs.lqg1d import LQG1D
from ifqi.algorithms.policy_gradient.policy import *
from ifqi.evaluation.evaluation import collect_episodes

mdp = LQG1D()
mu_b = -0.2
sigma_b = 2
mu_t = -0.2
sigma_t = 1
N = 100
H_min = 5
H_max = 20
learning_rate = 1e-4

#Target
target_policy = GaussianPolicyLinearMean(mu_t,sigma_t**2)

#Dataset
behavioral_policy = GaussianPolicyLinearMean(mu_b,sigma_b**2)
dataset = collect_episodes(mdp,behavioral_policy,n_episodes=N)

#Offline optimization
optimizer = HoeffdingOfflineLearner(H_min,
                                    H_max,
                                    dataset,
                                    behavioral_policy,
                                    target_policy,
                                    gamma=0.99,
                                    delta=0.01)
theta,history = optimizer.optimize(learning_rate,return_history=True,max_iter=100)
