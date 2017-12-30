from ifqi.algorithms.policy_gradient.offline_learner import *
from ifqi.envs.lqg1d import LQG1D
from ifqi.algorithms.policy_gradient.policy import *
from ifqi.evaluation.evaluation import collect_episodes
import matplotlib.pyplot as plt
import numpy as np

#Settings
mu_t = -0.2
sigma_t = 1.
H_min = 5
H_max = 20
learning_rate = 1e-4
target_policy = GaussianPolicyLinearMean(mu_t,sigma_t**2)
mu_b = -0.2
sigma_b = 2.
behavioral_policy = GaussianPolicyLinearMean(mu_b,sigma_b**2)

#Generate dataset
#print("Collecting dataset")
#mdp = LQG1D()
#N = 10000
#dataset = collect_episodes(mdp,behavioral_policy,n_episodes=N)

#Load dataset
print("Loading dataset")
dataset = np.load('dataset.npy')

#Offline optimization
print("Offline optimization")
optimizer = HoeffdingOfflineLearner(H_min,
                                    H_max,
                                    dataset,
                                    behavioral_policy,
                                    target_policy,
                                    gamma=0.99,
                                    delta=0.25)
theta,history = optimizer.optimize(learning_rate,
                                   learning_rate_search=False,
                                   return_history=True,
                                   max_iter=1000)

#Plotting
print("Plotting results")
history = np.array(history)
np.save('history.npy',history)
#plt.plot(range(history.shape[0]),history[:,0])
#plt.show()
