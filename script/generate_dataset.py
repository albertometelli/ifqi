import numpy as np
from ifqi.evaluation.evaluation import collect_episodes
from ifqi.envs.lqg1d import LQG1D
from ifqi.algorithms.policy_gradient.policy import *
import time
import json

#Settings
mdp = LQG1D()
mu_b = -0.2
sigma_b = 2.
N = 10000
behavioral_policy = GaussianPolicyLinearMean(mu_b,sigma_b**2)

#Generate dataset
print("Generating dataset...")
dataset = collect_episodes(mdp,behavioral_policy,n_episodes=N)

#Save dataset and info
print("Saving...")
dataset_path = '../datasets/'
dataset_label = str(N) #= time.strftime("%d-%m-%Y_%H-%M-%S")
np.save(dataset_path + 'dataset_' + dataset_label, dataset)
info = {'env' : 'LQG1D', 'mu_b' : mu_b, 'sigma_b' : sigma_b, 'N' : N}
with open(dataset_path + 'info_' + dataset_label + '.txt', 'w') as fp:
    fp.write(json.dumps(info))
