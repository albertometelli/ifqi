import copy
from spmi.envs import discrete
from spmi.utils.matrix_builders import *


class PathologicalCMDP(discrete.DiscreteEnv):
    def __init__(self, p, w, M):
        S = [0, 1, 2, 3, 4]
        A = [0]

        self.nS = len(S)
        self.nA = len(A)

        self.gamma = 0.99
        self.horizon = 5

        self.isd = np.zeros(self.nS)
        self.isd[0] = 0.5
        self.isd[3] = 0.5
        self.mu = self.isd

        self.P0 = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        self.P0[0][0].append((1 - p, 1, 1, False))
        self.P0[0][0].append((p, 4, 0, False))
        self.P0[1][0].append((p, 2, -M, False))
        self.P0[1][0].append((1 - p, 4, 0, False))
        self.P0[2][0].append((1 - p, 3, 1, False))
        self.P0[2][0].append((p, 4, 0, False))
        self.P0[3][0].append((1, 4, 0, False))
        self.P0[4][0].append((1, 4, 0, True))

        self.P1 = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        self.P1[0][0].append((p, 1, 1, False))
        self.P1[0][0].append((1 - p, 4, 0, False))
        self.P1[1][0].append((1 - p, 2, -M, False))
        self.P1[1][0].append((p, 4, 0, False))
        self.P1[2][0].append((p, 3, 1, False))
        self.P1[2][0].append((1 - p, 4, 0, False))
        self.P1[3][0].append((1, 4, 0, False))
        self.P1[4][0].append((1, 4, 0, True))

        self.P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        self.P[0][0].append((w * p + (1 - w) * (1 - p), 1, 1, False))
        self.P[0][0].append((w * (1 - p) + (1 - w) * p, 4, 0, False))
        self.P[1][0].append((w * (1 - p) + (1 - w) * p, 2, -M, False))
        self.P[1][0].append((w * p + (1 - w) * (1 - p), 4, 0, False))
        self.P[2][0].append((w * p + (1 - w) * (1 - p), 3, 1, False))
        self.P[2][0].append((w * (1 - p) + (1 - w) * p, 4, 0, False))
        self.P[3][0].append((1, 4, 0, False))
        self.P[4][0].append((1, 4, 0, True))

        self.P_sas = p_sas(self.P, self.nS, self.nA)
        self.P_sa = p_sa(self.P_sas, self.nS, self.nA)
        self.R_sas = r_sas(self.P, self.nS, self.nA)
        self.R = r_sa(self.R_sas, self.nS, self.nA)

        super(PathologicalCMDP, self).__init__(self.nS, self.nA, self.P, self.isd)

    def set_model(self, model):
        self.P = copy.deepcopy(model)
        self.P_sas = p_sas(self.P, self.nS, self.nA)
        self.P_sa = p_sa(self.P_sas, self.nS, self.nA)
        self.R_sas = r_sas(self.P, self.nS, self.nA)
        self.R = r_sa(self.R_sas, self.nS, self.nA)

    def get_valid_actions(self, s):
        return [0]

    # method to reset the MDP state to an initial one
    def reset(self):
        s = discrete.categorical_sample(self.isd, self.np_random)
        self.s = np.array([s]).ravel()
        return self.s

    # method to get the current MDP state
    def get_state(self):
        return self.s