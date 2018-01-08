import numpy as np
from scipy.optimize import fmin_slsqp


class SMC(object):

    def __init__(self, mdp, eps):
        """
        Safe Model Combinator:
        object that enable the call for smc methods

        :param mdp: mdp to solve
        :param eps: accuracy of the smc
        """
        self.mdp = mdp
        self.gamma = mdp.gamma
        self.horizon = mdp.horizon
        self.eps = eps


    # method to iterate safe model combination steps
    def safe_model_combination(self, policy, initial_w):

        # initializations
        mdp = self.mdp
        gamma = mdp.gamma
        Q_thresh = 0.0001
        d_mu_thresh = 0.0001

        k = len(initial_w)
        w = initial_w

        # compute U-function
        Q = self.model_q(policy, Q_thresh)
        U = self.model_u(policy, Q)

        # compute delta_mu
        d_mu = self.discounted_state_distribution(policy, d_mu_thresh)
        delta_mu = self.discounted_state_action_distribution(d_mu, policy)

        # compute vertex advantage vector
        X = self.vertex_advantage(delta_mu, U)

        # define the optimization objective function
        def obj_fun(w_next):
            dist = np.max(np.abs(w_next - w))
            a = dist ** 2
            b = k ** 2
            c = gamma ** 2
            d = (1 - gamma) ** 2
            num = a * b * c
            den = 2 * d
            penalty = ((dist ** 2) * (k ** 2) * (gamma ** 2)) / (2 * ((1 - gamma) ** 2))
            advantage = np.dot(w_next, X)
            fun = advantage - penalty
            return -fun

        # define the equality constraint (vector w sum to 1)
        def cons(w_next):
            return np.sum(w_next) - 1

        # # manual optimization
        # max = float("-inf")
        # for i in np.arange(0, 1, 1e-6):
        #     w_cur = np.array([i, 1 - i])
        #     score = obj_fun(w_cur)
        #     if score > max:
        #         max = score
        #         w_next = w_cur

        # define the optimization problem
        w_next = fmin_slsqp(obj_fun,
                            w,
                            eqcons=[cons],
                            bounds=[(0, 1)]*len(w))

        return w_next




    # method to compute the value of vertex model advantage (X)
    # X = sum(s)sum(a)[delta(sa) * sum(s')[PHI * U]]
    def vertex_advantage(self, delta_mu, U):

        # initializations
        mdp = self.mdp
        nS = mdp.nS
        nA = mdp.nA
        PHI = mdp.get_phi()
        k = len(PHI)

        X = np.zeros(k)
        # compute X[i] for each model in PHI
        for i in range(k):
            sum_sa = 0
            for sa in range(nS * nA):
                p_arr = PHI[i][sa]
                u_arr = U[sa]
                dot = np.dot(p_arr, u_arr)
                sum_sa = sum_sa + delta_mu[sa] * dot
            X[i] = sum_sa

        return X


    # method to exactly evaluate a model in terms of q-function
    # it returns Q: dictionary over states indexing array of values
    def model_q(self, policy, threshold):

        # initializations
        mdp = self.mdp
        nS = mdp.nS
        nA = mdp.nA
        P = mdp.P
        gamma = mdp.gamma
        policy_rep = policy.get_rep()

        # tabular q-function instantiation
        Q = {s: np.zeros(nA) for s in range(nS)}
        Qnext = {s: np.zeros(nA) for s in range(nS)}

        # evaluation loop
        delta = 1
        while delta > threshold:
            delta = 0
            for s in range(nS):
                valid = mdp.get_valid_actions(s)
                for a in valid:
                    sum = 0
                    temp = Q[s][a]
                    ns_list = P[s][a]
                    for elem in ns_list:
                        p = elem[0]
                        ns = elem[1]
                        r = elem[2]
                        pi_arr = policy_rep[ns]
                        q_arr = Q[ns]
                        next_ret = np.dot(pi_arr, q_arr)
                        sum = sum + p * (r + gamma * next_ret)
                    Qnext[s][a] = sum
                    delta = max(delta, np.abs(temp - sum))
            Q = Qnext

        return Q


    # method to compute the U-function
    def model_u(self, policy, Q):

        # initializations
        mdp = self.mdp
        nS = mdp.nS
        nA = mdp.nA
        gamma = mdp.gamma
        P_sa = mdp.P_sa
        R = mdp.R
        policy_rep = policy.get_rep()

        # U-function instantiation as (SA*S) matrix
        U = np.zeros(shape=(nS * nA, nS))

        # loop to fill the values in U
        for sa in range(nS * nA):
            for s1 in range(nS):
                # if the s1 is reachable from sa
                if P_sa[sa][s1] != 0:
                    q_arr = Q[s1]
                    pi_arr = policy_rep[s1]
                    V_s1 = np.dot(q_arr, pi_arr)
                    U[sa][s1] = R[s1] + gamma * V_s1

        return U


    # method to exactly compute the d_mu_P
    def discounted_state_distribution(self, policy, threshold):

        # initializations
        mdp = self.mdp
        nS = mdp.nS
        nA = mdp.nA
        gamma = mdp.gamma
        P_sa = mdp.P_sa
        mu = mdp.mu
        policy_rep = policy.get_rep()

        # d_mu as array with nS elements
        d_mu = np.zeros(nS)

        # transformation of policy_rep into a matrix SxSA
        pi = np.zeros(shape=(nS, nS * nA))
        a = 0
        s = 0
        for sa in range(nS * nA):
            if a == 5:
                a = 0
                s = s + 1
            pi[s][sa] = policy_rep[s][a]
            a = a + 1

        # computation of the SxS matrix P_pi
        P_pi = np.dot(pi, P_sa)

        # value iteration on d_mu
        delta = 1
        while delta > threshold:
            delta = 0
            dot = np.dot(d_mu, P_pi)
            d_mu_next = (1 - gamma) * mu + gamma * dot
            max_diff = max(np.absolute(d_mu - d_mu_next))
            delta = max(delta, max_diff)
            d_mu = d_mu_next

        return d_mu


    # method to compute the discounted state,action distribution
    def discounted_state_action_distribution(self, d_mu, policy):

        # initializations
        policy_rep = policy.get_rep()
        mdp = self.mdp
        nS = mdp.nS
        nA = mdp.nA

        # instantiation of delta_mu as SA vector
        delta_mu = np.zeros(nS * nA)

        # loop to fill the value in delta_mu
        a = 0
        s = 0
        for sa in range(nS * nA):
            if a == 5:
                a = 0
                s = s + 1
            delta_mu[sa] = d_mu[s] * policy_rep[s][a]
            a = a + 1

        return delta_mu
