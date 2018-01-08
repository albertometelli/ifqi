from ifqi.algorithms.policy_gradient.offline_learner import *
from ifqi.envs.lqg1d import LQG1D
from ifqi.algorithms.policy_gradient.policy import *
from ifqi.evaluation.evaluation import collect_episodes
import matplotlib.pyplot as plt
import numpy as np
import time
import json

#Settings
mu_t = -0.2
sigma_t = 1.
H_min = 4
H_max = 20
target_policy = GaussianPolicyLinearMean(mu_t,sigma_t**2)
gamma = .99
delta = .25
learning_rate_search = False
learning_rate = 1e-4 #fixed or initial learning rate
max_iter = 3
dataset_path = '../datasets/'
dataset_label = '10000'
batch_size = None
random_start = True

try:
    #Load dataset
    dataset = np.load(dataset_path + 'dataset_' + dataset_label + '.npy')
    with open(dataset_path + 'info_' + dataset_label + '.txt') as fp:
        info = json.load(fp)
    mu_b = info['mu_b']
    sigma_b = info['sigma_b']
    behavioral_policy = GaussianPolicyLinearMean(mu_b,sigma_b**2)
    N = info['N']
except:
    #Generate dataset
    print("Generating dataset")
    mdp = LQG1D()
    mu_b = -0.2
    sigma_b = 2.
    N = 10000
    behavioral_policy = GaussianPolicyLinearMean(mu_b,sigma_b**2)
    dataset = collect_episodes(mdp,behavioral_policy,n_episodes=N)

#Offline optimization
optimizer = HoeffdingOfflineLearner(H_min,
                                    H_max,
                                    dataset,
                                    behavioral_policy,
                                    target_policy,
                                    gamma=gamma,
                                    delta=delta,
                                    batch_size=batch_size,
                                    random_start=random_start)
theta,history = optimizer.optimize(learning_rate,
                                   learning_rate_search=learning_rate_search,
                                   return_history=True,
                                   max_iter=max_iter)

#Saving results
print("Saving results")
history = np.array(history)
history_path = '../results/'
history_label = 'example'
np.save(history_path + 'history_' + history_label, history)
info = {'env' : 'LQG1D', 
        'mu_b' : mu_b, 
        'sigma_b' : sigma_b, 
        'N' : N,
        'mu_t': mu_t,
        'sigma_t': sigma_t,
        'H_min': H_min,
        'delta': delta,
        'alpha': learning_rate
       }
with open(history_path + 'info_' + history_label + '.txt', 'w') as fp:
    fp.write(json.dumps(info))
