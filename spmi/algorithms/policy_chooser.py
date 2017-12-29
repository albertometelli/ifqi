import numpy as np
from spmi.utils import evaluator
from spmi.utils.tabular_operations import policy_sup_tv_distance, policy_mean_tv_distance
from spmi.utils.tabular import TabularPolicy

class PolicyChooser(object):

    def __init__(self, nS, nA):
        self.nS = nS
        self.nA = nA

    def choose(self, policy, d_mu_pi, Q):
        pass

class GreedyPolicyChooser(PolicyChooser):

    def choose(self, policy, d_mu_pi, Q):
        # GREEDY POLICY COMPUTATION
        target_policy_rep = self.greedy_policy(Q)
        # instantiation of a target policy object
        target_policy = TabularPolicy(target_policy_rep, self.nS, self.nA)

        # EXPECTED RELATIVE ADVANTAGE COMPUTATION
        er_advantage = evaluator.compute_policy_er_advantage(target_policy, policy, Q, d_mu_pi)

        # POLICY DISTANCE COMPUTATIONS
        distance_sup = policy_sup_tv_distance(target_policy, policy)
        distance_mean = policy_mean_tv_distance(target_policy, policy, d_mu_pi)

        return er_advantage, distance_sup, distance_mean, target_policy

    def greedy_policy(self, Q, tol=1e-3):

        greedy_policy_rep = {s: [] for s in range(self.nS)}
        # loop to give maximum probability to the greedy action,
        # if more than one is greedy then uniform on the greedy actions
        for s in range(self.nS):
            q_array = Q[s*self.nA : (s+1)*self.nA]
            probabilities = np.zeros(self.nA)

            # uniform if more than one greedy
            max = np.max(q_array)
            a = np.argwhere(np.abs(q_array - max) < tol).flatten()
            probabilities[a] = 1. / len(a)
            greedy_policy_rep[s] = probabilities

        return greedy_policy_rep

class SetPolicyChooser(PolicyChooser):

    def __init__(self, policy_set, nS, nA):
        self.policy_set = policy_set
        self.n_policies = len(self.policy_set)
        super(SetPolicyChooser, self).__init__(nS, nA)

    def choose(self, policy, d_mu_pi, Q):
        er_advantages = np.zeros(self.n_policies)

        for i in range(self.n_policies):
            er_advantages[i] = evaluator.compute_policy_er_advantage(self.policy_set[i], policy, Q, d_mu_pi)

        index = np.argmax(er_advantages)
        target_policy = self.policy_set[index]
        er_advantage = er_advantages[index]

        # POLICY DISTANCE COMPUTATIONS
        distance_sup = policy_sup_tv_distance(target_policy, policy)
        distance_mean = policy_mean_tv_distance(target_policy, policy, d_mu_pi)

        return er_advantage, distance_sup, distance_mean, target_policy