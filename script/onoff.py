from ifqi.envs.lqg1d import LQG1D
from ifqi.algorithms.policy_gradient.onoff_learner import OnOffLearner
import matplotlib.pyplot as plt
import numpy as np

mdp = LQG1D()
mdp.horizon = 10
sigma = 1.
mu = -0.2

learner = OnOffLearner(mdp,
                       mu,
                       sigma,
                       learn_sigma = True,
                       initial_batch_size=200,
                       batch_size_incr=10,
                       max_batch_size=3000,
                       select_initial_point=False,
                       adaptive_stop=True,
                       safe_stopping=True,
                       search_horizon=True,
                       adapt_batchsize=True,
                       bound='chebyshev',
                       delta=0.2,
                       importance_weighting_method='is',
                       learning_rate=0.002,
                       estimator='gpomdp',
                       gradient_updater='vanilla',
                       max_offline_iterations=50,
                       online_iterations=100,
                       verbose=1)

optimal_parameter, history, history_filter = learner.learn()
np.save('./history',history)
np.save('./history_filter',history_filter)
