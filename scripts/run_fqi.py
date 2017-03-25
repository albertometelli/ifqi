import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor

from ifqi import envs
from ifqi.evaluation import evaluation
from ifqi.fqi.FQI import FQI
from ifqi.models.actionregressor import ActionRegressor
from ifqi.models.mlp import MLP
from ifqi.models.ensemble import ExtraTreesEnsemble

"""
Simple script to quickly run fqi. It solves the Acrobot environment according
to the experiment presented in:

Ernst, Damien, Pierre Geurts, and Louis Wehenkel.
"Tree-based batch mode reinforcement learning."
Journal of Machine Learning Research 6.Apr (2005): 503-556.
"""

mdp = envs.Acrobot()

# mdp.set_seed(seed)
state_dim, action_dim, reward_dim = envs.get_space_info(mdp)
regressor_params = {'n_estimators': 50,
                    'criterion': 'mse',
                    'min_samples_split': 5,
                    'min_samples_leaf': 2}
discrete_actions = mdp.action_space.values
regressor = ExtraTreesRegressor(**regressor_params)
# regressor = MLP(3, 1, [15], 'relu', 'rmsprop')

regressor = ActionRegressor(regressor, discrete_actions=discrete_actions,
                            decimals=5, **regressor_params)

dataset = evaluation.collect_episodes(mdp, n_episodes=2000)
print('Dataset has %d samples' % dataset.shape[0])

reward_idx = state_dim + action_dim
sast = np.append(dataset[:, :reward_idx], dataset[:, reward_idx + 1:], axis=1)
r = dataset[:, reward_idx]

fqi = FQI(estimator=regressor,
          state_dim=state_dim,
          action_dim=action_dim,
          discrete_actions=discrete_actions,
          gamma=mdp.gamma,
          horizon=mdp.horizon,
          scaled=False,
          features=None,
          verbose=True)

fitParams = {}

initial_states = np.zeros((41, 4))
initial_states[:, 0] = np.linspace(-2, 2, 41)

fqi.partial_fit(sast, r, **fitParams)

iterations = 100
n_test_episodes = initial_states.shape[0]
iteration_values = []
for i in range(iterations - 1):
    fqi.partial_fit(None, None, **fitParams)
    if not i % 5:
        values = evaluation.evaluate_policy(mdp, fqi,
                                            initial_states=initial_states,
                                            n_episodes=n_test_episodes)
        iteration_values.append(values[0])

        if i == 1:
            fig1 = plt.figure(1)
            ax = fig1.add_subplot(1, 1, 1)
            h = ax.plot(range(i + 1), iteration_values, 'ro-')
            plt.ylim(min(iteration_values), max(iteration_values))
            plt.xlim(0, i + 1)
            plt.ion()  # turns on interactive mode
            plt.show()
        elif i > 1:
            h[0].set_data(range(i + 1), iteration_values)
            ax.figure.canvas.draw()
            plt.ylim(min(iteration_values), max(iteration_values))
            plt.xlim(0, i + 1)
            plt.show()
