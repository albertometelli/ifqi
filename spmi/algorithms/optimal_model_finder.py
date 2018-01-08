import numpy as np


class OptimalModelFinder(object):

    def __init__(self, mdp):

        self.mdp = mdp
        self.coefficient = list()
        self.performance = list()

    def optimal_model_finder(self, grid_step, threshold):

        # initializations
        mdp = self.mdp
        nS = mdp.nS
        gamma = mdp.gamma
        mu = mdp.mu
        R = mdp.R

        # value iteration function
        def value_iteration():

            V = np.zeros(nS)
            V_next = np.zeros(nS)
            P_sas = mdp.P_sas
            delta = float("inf")

            while delta > threshold:
                delta = 0
                V_sa = np.dot(P_sas, R + gamma * V)
                for s in range(nS):
                    V_next[s] = np.max(V_sa[s])
                max_diff = max(np.absolute(V_next - V))
                delta = max(delta, max_diff)
                V = np.copy(V_next)

            return V

        # loop performance computation
        for i in np.arange(0, 1, grid_step):
            mdp.model_configuration(i)
            V_star = value_iteration()
            J_star = np.dot(mu, V_star)
            self.coefficient.append(i)
            self.performance.append(J_star)
            print("iteration: {0} of {1}".format(i, 1 / grid_step))

        return
