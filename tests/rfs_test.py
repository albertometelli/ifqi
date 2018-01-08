from __future__ import print_function
import spmi.envs as env
from spmi.envs.utils import get_space_info
from spmi.evaluation import evaluation
from spmi.evaluation.utils import check_dataset, split_dataset
from spmi.algorithms.selection import RFS, IFS
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np

from sklearn.feature_selection import RFE

# np.random.seed(3452)

mdp = env.SyntheticToyFS()
state_dim, action_dim, reward_dim = get_space_info(mdp)
nextstate_idx = state_dim + action_dim + reward_dim
reward_idx = action_dim + state_dim

# dataset: s, a, r, s'
dataset = evaluation.collect_episodes(mdp, n_episodes=50)
check_dataset(dataset, state_dim, action_dim, reward_dim)

selector = IFS(estimator=ExtraTreesRegressor(n_estimators=50),
               scale=True, verbose=1)
fs = RFS(feature_selector=selector,
         features_names=np.array(['S0', 'S1', 'S2', 'S3', 'A0', 'A1']),
         verbose=1)

state, actions, reward, next_states, absorbing = \
    split_dataset(dataset, state_dim, action_dim, reward_dim)

# print(dataset[:10, :])

fs.fit(state, actions, next_states, reward)
selected_features = fs.features_names[fs.get_support()]
print('selected features: {}'.format(selected_features))  # this are the selected features, it should be [s0, s2, a0]
assert np.all(selected_features == ['S0', 'S2', 'A0'])

print(fs.nodes)
g = fs.export_graphviz()
g.view()
