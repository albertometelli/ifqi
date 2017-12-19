from ifqi.envs.lqg1d import LQG1D
from ifqi.algorithms.policy_gradient.policy import GaussianPolicyMean
from ifqi.algorithms.policy_gradient.policy_gradient_learner import PolicyGradientLearner
import matplotlib.pyplot as plt
import numpy as np

mdp = LQG1D()
K_opt = mdp.computeOptimalK()

st = 1

#Instantiate policies
target = GaussianPolicyMean(0., st**2)

reinforce = PolicyGradientLearner(mdp,
                                  target,
                                  lrate=0.005,
                                  estimator='reinforce',
                                  gradient_updater='adam',
                                  max_iter_opt=100,
                                  max_iter_eval=100,
                                  verbose=1)

gpomdp = PolicyGradientLearner(mdp,
                                  target,
                                  lrate=0.005,
                                  estimator='gpomdp',
                                  gradient_updater='adam',
                                  max_iter_opt=100,
                                  max_iter_eval=100,
                                  verbose=1)

theta0 = target.from_param_to_vec(-0.2)
_, history_reinforce = reinforce.optimize(theta0, return_history=True)
_, history_gpomdp = gpomdp.optimize(theta0, return_history=True)

fig, ax = plt.subplots()
ax.plot(np.array(history_reinforce)[:, 1], 'r', label='Reinforce')
ax.plot(np.array(history_gpomdp)[:, 1], 'b--', label='GPOMDP')
legend = ax.legend(loc='lower right')