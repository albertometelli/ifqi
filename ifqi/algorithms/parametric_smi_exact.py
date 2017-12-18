import numpy as np


class pSMI(object):

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

        # attributes related to
        # trace print procedures
        self.count = 0
        self.iteration = 0
        self.iterations = list()
        self.evaluations = list()
        self.advantages = list()
        self.distances_sup = list()
        self.distances_mean = list()
        self.coefficients = list()
        self.betas = list()


    # method to iterate safe model combination steps
    def parametric_safe_model_iteration(self, policy, initial_w):

        # initializations
        mdp = self.mdp
        gamma = mdp.gamma
        U_thresh = 0.0001
        delta_mu_thresh = 0.0001
        eps = self.eps
        iteration_horizon = 2000

        # update model until convergence
        w = initial_w
        U = self.model_u(policy, U_thresh)
        delta_mu = self.discounted_sa_distribution(policy, delta_mu_thresh)
        er_advantage, distance_sup, distance_mean, target_index = self.model_chooser(w, delta_mu, U)
        target_index_old = target_index
        convergence_condition = eps / (1 - gamma)
        while er_advantage > convergence_condition and self.iteration < iteration_horizon:
            beta_num = (1 - gamma) * er_advantage
            beta_den = 2 * (gamma ** 2) * distance_sup * distance_mean
            beta_star = beta_num / beta_den
            beta = min(1, beta_star)
            w = self.model_combination(w, beta, target_index)
            U = self.model_u(policy, U_thresh)
            delta_mu = self.discounted_sa_distribution(policy, delta_mu_thresh)
            J_P = self.model_performance(policy, U)

            self._utility_trace(J_P, beta_star, er_advantage, convergence_condition,
                                distance_sup, distance_mean, w, target_index, target_index_old)

            target_index_old = target_index
            er_advantage, distance_sup, distance_mean, target_policy = self.model_chooser(w, delta_mu, U)

        return w


    # method which returns the target model,
    # er advantage, mean and sup distance
    def model_chooser(self, w, delta_mu, U):

        # initializations
        mdp = self.mdp
        PHI = mdp.get_phi()
        P_sa = mdp.P_sa
        k = len(w)

        # compute er_advantage for each model
        vertex_adv = self.vertex_advantage(delta_mu, U)

        # select the target model and target adv
        target_index = np.argmax(vertex_adv)
        er_advantage = vertex_adv[target_index]

        # # compute the distances
        # w_target = np.zeros(k)
        # w_target[target_index] = 1
        # distance_sup = np.max(np.abs(w_target - w))
        # distance_sup = (k ** 2) * distance_sup

        # compute the distances
        P_target = PHI[target_index]
        distance_sup = self.model_infinite_norm(P_target, P_sa)
        distance_mean = self.model_mean_distance(P_target, P_sa, delta_mu)

        return er_advantage, distance_sup, distance_mean, target_index


    # method to compute the infinite norm between two given models
    def model_infinite_norm(self, P1_sa, P2_sa):

        # initializations
        mdp = self.mdp
        nS = mdp.nS
        nA = mdp.nA

        # infinite norm computation
        sum_sa = np.zeros(nS * nA)
        for sa in range(nS * nA):
            diff_s = P1_sa[sa] - P2_sa[sa]
            sum_sa[sa] = np.sum(np.absolute(diff_s))
        max = np.max(sum_sa)

        return max


    # method to compute the mean value distance between two given models
    def model_mean_distance(self, P1_sa, P2_sa, delta_mu):

        # initializations
        mdp = self.mdp
        nA = mdp.nA
        nS = mdp.nS

        # mean value computation
        sum_sa = np.zeros(nS * nA)
        for sa in range(nS * nA):
            diff_s = P1_sa[sa] - P2_sa[sa]
            sum_sa[sa] = np.sum(np.absolute(diff_s))
        mean = np.dot(delta_mu, sum_sa)

        return mean


    # method which compute the new combination
    # coefficients and calls the model_configuration()
    def model_combination(self, w, beta, target_index):

        # initializations
        mdp = self.mdp

        # coefficients computation
        w_next = w * (1 - beta)
        w_next[target_index] = beta + w[target_index] * (1 - beta)

        # model_configuration call
        mdp.model_configuration(w_next[0])

        return w_next


    # method which compute the new model performance
    def model_performance(self, policy, U):

        # initializations
        policy_rep = policy.get_rep()
        mdp = self.mdp
        mu = mdp.mu
        nS = mdp.nS

        # Q-function computation
        Q = self.model_q(policy, 0.0001)

        # V-function computation
        V = np.zeros(nS)
        for s in range(nS):
            pi_arr = policy_rep[s]
            q_arr = Q[s]
            V[s] = np.dot(pi_arr, q_arr)

        # performance computation
        J_P = np.dot(mu, V)

        ###########################
        # initializations
        # v-function computation
        # performance computation

        return J_P


    # method to compute the value of vertex models advantage
    def vertex_advantage(self, delta_mu, U):

        # initializations
        mdp = self.mdp
        nS = mdp.nS
        nA = mdp.nA
        PHI = mdp.get_phi()
        P_sa = mdp.P_sa
        k = len(PHI)

        vertex_adv = np.zeros(k)
        # compute vertex_adv[i] for each model in PHI
        for i in range(k):
            sum_sa = 0
            for sa in range(nS * nA):
                p_i_arr = PHI[i][sa]
                p_arr = P_sa[sa]
                diff_arr = p_i_arr - p_arr
                u_arr = U[sa]
                dot = np.dot(diff_arr, u_arr)
                sum_sa = sum_sa + delta_mu[sa] * dot
            vertex_adv[i] = sum_sa

        return vertex_adv


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
    def model_u(self, policy, U_thresh):

        # initializations
        mdp = self.mdp
        nS = mdp.nS
        nA = mdp.nA
        gamma = mdp.gamma
        P_sa = mdp.P_sa
        R = mdp.R
        policy_rep = policy.get_rep()

        # Q-function computation
        Q = self.model_q(policy, U_thresh)

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
    def discounted_s_distribution(self, policy, threshold):

        # initializations
        mdp = self.mdp
        nS = mdp.nS
        nA = mdp.nA
        gamma = mdp.gamma
        P_sa = mdp.P_sa
        mu = mdp.mu
        h = mdp.horizon
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

        # h-iterations on d_mu to have d_mu_h
        count = 0
        while count < h:
            dot = np.dot(d_mu, P_pi)
            d_mu_next = (1 - gamma) * mu + gamma * dot
            d_mu = d_mu_next
            count = count + 1

        return d_mu


    # method to compute the discounted state,action distribution
    def discounted_sa_distribution(self, policy, delta_mu_thresh):

        # initializations
        policy_rep = policy.get_rep()
        mdp = self.mdp
        nS = mdp.nS
        nA = mdp.nA

        # d_mu computation
        d_mu = self.discounted_s_distribution(policy, delta_mu_thresh)

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


    # utility method to print the algorithm trace
    # and to collect execution data
    def _utility_trace(self, J_P, beta_star, er_advantage, convergence_condition,
                       distance_sup, distance_mean, w, target_index, target_index_old):

        # data collections
        self.iterations.append(self.iteration)
        self.coefficients.append(w[0])
        self.evaluations.append(J_P)
        self.betas.append(beta_star)
        self.advantages.append(er_advantage)
        self.distances_sup.append(distance_sup)
        self.distances_mean.append(distance_mean)
        average = np.mean(self.evaluations)

        # 10-moving average computation
        moving_average = 0
        moving_count = self.iteration
        while moving_count >= 0 and moving_count >= (self.iteration - 9):
            moving_average = moving_average + self.evaluations[moving_count]
            moving_count = moving_count - 1
        moving_average = moving_average / 10

        # target change check
        check_target = (target_index == target_index_old)

        # trace print
        print('----------------------')
        print('Model: {0}\n'.format(w))
        print('Beta star: {0}'.format(beta_star))
        print('Distance sup: {0}'.format(distance_sup))
        print('Distance mean: {0}'.format(distance_mean))
        print('Condition: {0}'.format(convergence_condition))
        print('Advantage: {0}'.format(er_advantage))
        print('Same target: {0}\n'.format(check_target))
        print('Model evaluation: {0}'.format(J_P))
        print('Evaluation average: {0}'.format(average))
        print('Evaluation moving average (10): {0}'.format(moving_average))
        print('Iteration: {0}'.format(self.iteration))

        # iteration update
        self.iteration = self.iteration + 1
