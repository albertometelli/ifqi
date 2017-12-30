import numpy as np
from ifqi.evaluation.evaluation import collect_episodes
from ifqi.envs.lqg1d import LQG1D
from ifqi.algorithms.policy_gradient.policy import *

#Settings
mdp = LQG1D()
mu_b = -0.2
sigma_b = 2.0
N = 10000
behavioral_policy = GaussianPolicyLinearMean(mu_b,sigma_b**2)

#Generate dataset
print("Generating dataset")
dataset = collect_episodes(mdp,behavioral_policy,n_episodes=N)
np.save('dataset.npy',dataset)
print("Done")
