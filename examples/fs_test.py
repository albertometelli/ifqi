from __future__ import print_function
import spmi.envs as env
from spmi.envs.utils import get_space_info
from spmi.evaluation import evaluation
from spmi.evaluation.utils import check_dataset, split_dataset
from spmi.algorithms.selection import RFS, IFS
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np

# np.random.seed(3452)

mdp = env.GridWorldEnv()
state_dim, action_dim, reward_dim = get_space_info(mdp)
state_dim = 49
action_dim = 1
reward_dim = 1
nextstate_idx = state_dim + action_dim + reward_dim
reward_idx = action_dim + state_dim

# dataset: s, a, r, s'
# dataset = evaluation.collect_episodes(mdp, n_episodes=50)
dataset = np.loadtxt('encoded_dataset.csv', skiprows=1, delimiter=',')
# check_dataset(dataset, state_dim, action_dim, reward_dim)

estimator = ExtraTreesRegressor(n_estimators=50, n_jobs=-1,
                                             importance_criterion="gini")
# estimator = DecisionTreeRegressor(importance_criterion="gini")

selector = IFS(estimator=estimator,
               scale=True, verbose=1)
features_names = ['S%s' % i for i in xrange(state_dim)] + ['A%s' % i for i in
                                                           xrange(action_dim)]
fs = RFS(feature_selector=selector,
         # features_names=np.array(['S0', 'S1', 'S2', 'S3', 'A0', 'A1']),
         features_names=np.array(features_names),
         verbose=1)

state, actions, reward, next_states = \
    split_dataset(dataset, state_dim, action_dim, reward_dim)

state = dataset[:,0:state_dim]
actions = dataset[:,state_dim:state_dim+action_dim]
reward = dataset[:,state_dim+action_dim]

# print(dataset[:10, :])

fs.fit(state, actions, next_states, reward)
print(
    fs.get_support())  # this are the selected features, it should be [s0, s2, a0]
