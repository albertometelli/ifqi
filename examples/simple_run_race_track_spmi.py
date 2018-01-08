from spmi.envs.race_track_configurable import RaceTrackConfigurableEnv
from spmi.evaluation.evaluation import collect_episodes
from spmi.algorithms.spmi_exact_non_par import SPMI
from spmi.utils.uniform_policy import UniformPolicy
from spmi.utils.tabular import TabularPolicy, TabularModel
from spmi.algorithms.policy_chooser import *
from spmi.algorithms.model_chooser import *
import numpy as np
from gym.utils import seeding
import matplotlib.pyplot as plt

mdp = RaceTrackConfigurableEnv(track_file='track0', initial_configuration=0.5)
mdp._render()
uniform_policy = UniformPolicy(mdp)

initial_model = TabularModel(mdp.P, mdp.nS, mdp.nA)
initial_policy = TabularPolicy(uniform_policy.get_rep(), mdp.nS, mdp.nA)

model_set = [TabularModel(mdp.P1, mdp.nS, mdp.nA), TabularModel(mdp.P2, mdp.nS, mdp.nA)]

policy_chooser = GreedyPolicyChooser(mdp.nS, mdp.nA)
model_chooser = SetModelChooser(model_set, mdp.nS, mdp.nA)

eps = 0.0
spmi = SPMI(mdp, eps, policy_chooser, model_chooser, max_iter=3000, use_target_trick=True, delta_q=1)

#-------------------------------------------------------------------------------
#SPMI
policy, model = spmi.safe_policy_model_iteration(initial_policy, initial_model)

iterations = np.array(range(spmi.iteration))
evaluations = np.array(spmi.evaluations)
p_advantages = np.array(spmi.p_advantages)
m_advantages = np.array(spmi.m_advantages)
p_dist_sup = np.array(spmi.p_dist_sup)
p_dist_mean = np.array(spmi.p_dist_mean)
m_dist_sup = np.array(spmi.m_dist_sup)
m_dist_mean = np.array(spmi.m_dist_mean)
alfas = np.array(spmi.alfas)
betas = np.array(spmi.betas)
p_change = np.cumsum(1 - np.array(spmi.p_change))
m_change = np.cumsum(1 - np.array(spmi.m_change))