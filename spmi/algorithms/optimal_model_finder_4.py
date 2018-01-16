import numpy as np
from sklearn.model_selection import ParameterGrid
from spmi.utils.tabular import *


class OptimalModelFinder4(object):

    def __init__(self, mdp, model_set):

        self.mdp = mdp
        self.model_set = model_set
        self.coefficient = list()
        self.performance = list()

    def optimal_model_finder(self, grid_num, threshold):

        # initializations
        mu = self.mdp.mu
        gamma = self.mdp.gamma
        nS, nA = self.mdp.nS, self.mdp.nA
        reward = self.mdp.R

        P0 = self.model_set[0].get_matrix()
        P1 = self.model_set[1].get_matrix()
        P2 = self.model_set[2].get_matrix()
        P3 = self.model_set[3].get_matrix()

        # value iteration function
        def value_iteration(P):

            V = np.zeros(nS)
            V_next = np.zeros(nS)
            P_sa_s = P
            delta = float("inf")

            while delta > threshold:
                delta = 0
                V_sa = np.dot(P_sa_s, reward + gamma * V)
                for s in range(nS):
                    V_next[s] = np.max(V_sa[s * nA: s*nA + nA])
                max_diff = max(np.absolute(V_next - V))
                delta = max(delta, max_diff)
                V = np.copy(V_next)

            return V

        # convex combination function
        def convex_combination(comb):
            return comb[0] * P0 + comb[1] * P1 + comb[2] * P2 + comb[3] * P3

        # grid generation (not_convex)
        lin = np.linspace(0., 1., num=grid_num, endpoint=True)
        param_grid = {'a': lin, 'b': lin, 'c': lin, 'd': lin}
        grid = list(ParameterGrid(param_grid))

        # grid filtering into a convex one
        convex_grid = list()
        for elem in grid:
            comb = elem['a'] + elem['b'] + elem['c'] + elem['d']
            if comb == 1:
                convex_grid.append([elem['a'], elem['b'], elem['c'], elem['d']])

        i = 0
        J_star = 0
        Js = []
        comb_star = [0, 0, 0, 0]
        # loop performance computation

        a = np.linspace(0., 1., 11)
        convex_grid = zip(a, np.zeros(11), 1. - a, np.zeros(11))

        for comb in convex_grid:
            P = convex_combination(comb)
            V = value_iteration(P)
            J = np.dot(mu, V)
            Js.append(J)
            if J > J_star:
                J_star = J
                comb_star = comb
            print("iteration: {0} of {1} - {2}".format(i, len(convex_grid), J))
            i = i + 1

        return comb_star, convex_grid, Js
