import numpy as np
from spmi.utils.tictoc import tic, toc
from spmi.utils.uniform_policy import UniformPolicy
from spmi.utils.spi_policy import SpiPolicy


class SPMI(object):

    def __init__(self, mdp, eps, max_iter=10000, delta_q=None, use_target_trick=True):
        """
        Safe Policy Model Iterator:
        object that enable the call for exact spmi algorithms

        :param mdp: mdp to solve
        :param eps: accuracy of the spi
        """
        self.mdp = mdp
        self.gamma = mdp.gamma
        self.horizon = mdp.horizon
        self.eps = eps
        self.iteration_horizon = max_iter
        self.threshold = 0.01
        self.use_target_trick = use_target_trick

        if delta_q is None:
            self.delta_q = (1. - self.gamma ** self.horizon) / (1 - self.gamma)
        else:
            self.delta_q = delta_q

        # attributes related to
        # trace print procedures
        self.count = 0
        self.iteration = 0
        self.iterations = list()
        self.evaluations = list()
        self.p_advantages = list()
        self.m_advantages = list()
        self.p_dist_sup = list()
        self.p_dist_mean = list()
        self.m_dist_sup = list()
        self.m_dist_mean = list()
        self.alfas = list()
        self.betas = list()
        self.coefficients = list()


    # implementation of safe policy (and) model iteration
    def safe_policy_model_iteration(self, initial_policy, initial_model):

        # initializations
        gamma = self.gamma
        eps = self.eps
        thresh = self.threshold
        iteration_horizon = self.iteration_horizon
        self._reset_trace()


        # policy chooser
        policy = initial_policy
        Q = self.policy_q(policy, thresh)
        d_mu = self.discounted_s_distribution(policy)
        p_er_adv, p_dist_sup, p_dist_mean, target = self.policy_chooser(policy, d_mu, Q)
        target_old = target

        # model chooser
        model = initial_model
        U = self.model_u(policy, thresh, Q)
        delta_mu = self.discounted_sa_distribution(policy, d_mu)
        m_er_adv, m_dist_sup, m_dist_mean, target_index = self.model_chooser(model, delta_mu, U)
        target_index_old = target_index


        # check convergence condition
        convergence = eps / (1 - gamma)
        while ((p_er_adv + m_er_adv) > convergence) and self.iteration < iteration_horizon:

            target_to_test = [(target, p_er_adv, p_dist_sup, p_dist_mean)]
            if self.use_target_trick:
                if not self.policy_equiv_check(target, target_old) and not self.policy_equiv_check(policy, target_old):
                    er_adv_old, dist_sup_old, dist_mean_old = self.policy_chooser_old(policy, target_old, d_mu, Q)
                    target_to_test.append((target_old, er_adv_old, dist_sup_old, dist_mean_old))

            bound_star = 0.
            alpha_star = 0.
            beta_star = 0.
            target_star = target

            for target, p_er_adv, p_dist_sup, p_dist_mean in target_to_test:

                alpha0 = ((1 - gamma) * p_er_adv) / (self.delta_q * gamma * \
                            p_dist_sup * p_dist_mean + 1e-24)
                alpha1 = ((1 - gamma) * p_er_adv) / (self.delta_q * gamma * \
                            p_dist_sup * p_dist_mean + 1e-24) - .5 * \
                            (m_dist_mean / (p_dist_mean + 1e-24) + m_dist_sup / (p_dist_sup + 1e-24))
                beta0 = ((1 - gamma) * m_er_adv) / (self.delta_q * (gamma ** 2) \
                            * m_dist_sup * m_dist_mean)
                beta1 = ((1 - gamma) * m_er_adv) / (self.delta_q * (gamma ** 2) \
                            * m_dist_sup * m_dist_mean) - .5 / gamma * \
                            (p_dist_mean / (m_dist_mean + 1e-24) + p_dist_sup / (m_dist_sup + 1e-24))

                alpha0 = np.clip(alpha0, 0., 1.)
                alpha1 = np.clip(alpha1, 0., 1.)
                beta0 = np.clip(beta0, 0., 1.)
                beta1 = np.clip(beta1, 0., 1.)

                for alpha, beta in [(alpha0, 0.), (0., beta0), (alpha1, 1.), (1., beta1)]:

                    bound = alpha * p_er_adv + beta * m_er_adv - \
                            (gamma / (1 - gamma) * self.delta_q / 2) * \
                            ((alpha ** 2) * p_dist_sup * p_dist_mean + \
                             gamma * (beta ** 2) * m_dist_sup * m_dist_mean + \
                             alpha * beta * p_dist_sup * m_dist_mean + alpha * beta * p_dist_mean * m_dist_sup)

                    if bound > bound_star:
                        bound_star = bound
                        alpha_star = alpha
                        beta_star = beta
                        target_star = target

            # policy and model update
            if alpha_star > 0:
                policy = self.policy_combination(alpha_star, target_star, policy)
            if beta_star > 0:
                model = self.model_combination(beta_star, target_index, model)

            # performance evaluation
            Q = self.policy_q(policy, thresh)
            J_p_m = self.performance(policy, Q)

            self._utility_trace(J_p_m, alpha_star, beta_star, p_er_adv, m_er_adv,
                                p_dist_sup, p_dist_mean, m_dist_sup, m_dist_mean,
                                target_star, target_old, target_index, target_index_old, convergence, model)

            # policy chooser
            target_old = target_star
            d_mu = self.discounted_s_distribution(policy)
            p_er_adv, p_dist_sup, p_dist_mean, target = self.policy_chooser(policy, d_mu, Q)

            # model chooser
            target_index_old = target_index
            U = self.model_u(policy, thresh, Q)
            delta_mu = self.discounted_sa_distribution(policy, d_mu)
            m_er_adv, m_dist_sup, m_dist_mean, target_index = self.model_chooser(model, delta_mu, U)

        return policy, model

    # implementation of safe policy (and) model iteration
    # version with sup distances in alfa, beta computation
    def safe_policy_model_iteration_sup(self, initial_policy, initial_model):

        # initializations
        gamma = self.gamma
        eps = self.eps
        thresh = self.threshold
        iteration_horizon = self.iteration_horizon
        self._reset_trace()


        # policy chooser
        policy = initial_policy
        Q = self.policy_q(policy, thresh)
        d_mu = self.discounted_s_distribution(policy)
        p_er_adv, p_dist_sup, p_dist_mean, target = self.policy_chooser(policy, d_mu, Q)
        target_old = target

        # model chooser
        model = initial_model
        U = self.model_u(policy, thresh, Q)
        delta_mu = self.discounted_sa_distribution(policy, d_mu)
        m_er_adv, m_dist_sup, m_dist_mean, target_index = self.model_chooser(model, delta_mu, U)
        target_index_old = target_index


        # check convergence condition
        convergence = eps / (1 - gamma)
        while ((p_er_adv + m_er_adv) > convergence) and self.iteration < iteration_horizon:

            target_to_test = [(target, p_er_adv, p_dist_sup, p_dist_mean)]
            if self.use_target_trick:
                if not self.policy_equiv_check(target, target_old) and not self.policy_equiv_check(policy, target_old):
                    er_adv_old, dist_sup_old, dist_mean_old = self.policy_chooser_old(policy, target_old, d_mu, Q)
                    target_to_test.append((target_old, er_adv_old, dist_sup_old, dist_mean_old))

            bound_star = 0.
            alpha_star = 0.
            beta_star = 0.
            target_star = target

            for target, p_er_adv, p_dist_sup, p_dist_mean in target_to_test:

                alpha0 = ((1 - gamma) * p_er_adv) / (self.delta_q * gamma * \
                            p_dist_sup ** 2 + 1e-24)
                alpha1 = ((1 - gamma) * p_er_adv) / (self.delta_q * gamma * \
                            p_dist_sup ** 2 + 1e-24) - \
                            m_dist_sup / (p_dist_sup + 1e-24)
                beta0 = ((1 - gamma) * m_er_adv) / (self.delta_q * (gamma ** 2) \
                            * m_dist_sup ** 2)
                beta1 = ((1 - gamma) * m_er_adv) / (self.delta_q * (gamma ** 2) \
                            * m_dist_sup ** 2) - 1. / gamma * \
                            p_dist_sup / (m_dist_sup + 1e-24)

                alpha0 = np.clip(alpha0, 0., 1.)
                alpha1 = np.clip(alpha1, 0., 1.)
                beta0 = np.clip(beta0, 0., 1.)
                beta1 = np.clip(beta1, 0., 1.)

                for alpha, beta in [(alpha0, 0.), (0., beta0), (alpha1, 1.), (1., beta1)]:

                    bound = alpha * p_er_adv + beta * m_er_adv - \
                            (gamma / (1 - gamma) * self.delta_q / 2) * \
                            ((alpha ** 2) * p_dist_sup ** 2 + \
                             gamma * (beta ** 2) * m_dist_sup ** 2 + \
                             2 * alpha * beta * p_dist_sup * m_dist_sup)

                    if bound > bound_star:
                        bound_star = bound
                        alpha_star = alpha
                        beta_star = beta
                        target_star = target

            # policy and model update
            if alpha_star > 0:
                policy = self.policy_combination(alpha_star, target_star, policy)
            if beta_star > 0:
                model = self.model_combination(beta_star, target_index, model)

            # performance evaluation
            Q = self.policy_q(policy, thresh)
            J_p_m = self.performance(policy, Q)

            self._utility_trace(J_p_m, alpha_star, beta_star, p_er_adv, m_er_adv,
                                p_dist_sup, p_dist_mean, m_dist_sup, m_dist_mean,
                                target_star, target_old, target_index, target_index_old, convergence, model)

            # policy chooser
            target_old = target_star
            d_mu = self.discounted_s_distribution(policy)
            p_er_adv, p_dist_sup, p_dist_mean, target = self.policy_chooser(policy, d_mu, Q)

            # model chooser
            target_index_old = target_index
            U = self.model_u(policy, thresh, Q)
            delta_mu = self.discounted_sa_distribution(policy, d_mu)
            m_er_adv, m_dist_sup, m_dist_mean, target_index = self.model_chooser(model, delta_mu, U)

        return policy, model

    # implementation of safe policy (and) model iteration
    # version which avoids full step to the target policy (model)
    def safe_policy_model_iteration_no_full_step(self, initial_policy, initial_model):

        # initializations
        gamma = self.gamma
        eps = self.eps
        thresh = self.threshold
        iteration_horizon = self.iteration_horizon
        self._reset_trace()


        # policy chooser
        policy = initial_policy
        Q = self.policy_q(policy, thresh)
        d_mu = self.discounted_s_distribution(policy)
        p_er_adv, p_dist_sup, p_dist_mean, target = self.policy_chooser(policy, d_mu, Q)
        target_old = target

        # model chooser
        model = initial_model
        U = self.model_u(policy, thresh, Q)
        delta_mu = self.discounted_sa_distribution(policy, d_mu)
        m_er_adv, m_dist_sup, m_dist_mean, target_index = self.model_chooser(model, delta_mu, U)
        target_index_old = target_index


        # check convergence condition
        convergence = eps / (1 - gamma)
        while ((p_er_adv + m_er_adv) > convergence) and self.iteration < iteration_horizon:

            target_to_test = [(target, p_er_adv, p_dist_sup, p_dist_mean)]
            if self.use_target_trick:
                if not self.policy_equiv_check(target, target_old) and not self.policy_equiv_check(policy, target_old):
                    er_adv_old, dist_sup_old, dist_mean_old = self.policy_chooser_old(policy, target_old, d_mu, Q)
                    target_to_test.append((target_old, er_adv_old, dist_sup_old, dist_mean_old))

            bound_star = 0.
            alpha_star = 0.
            beta_star = 0.
            target_star = target

            for target, p_er_adv, p_dist_sup, p_dist_mean in target_to_test:

                alpha0 = ((1 - gamma) * p_er_adv) / (self.delta_q * gamma * \
                            p_dist_sup * p_dist_mean)
                beta0 = ((1 - gamma) * m_er_adv) / (self.delta_q * (gamma ** 2) \
                            * m_dist_sup * m_dist_mean)

                alpha0 = np.clip(alpha0, 0., 1.)
                beta0 = np.clip(beta0, 0., 1.)

                for alpha, beta in [(alpha0, 0.), (0., beta0)]:

                    bound = alpha * p_er_adv + beta * m_er_adv - \
                            (gamma / (1 - gamma) * self.delta_q / 2) * \
                            ((alpha ** 2) * p_dist_sup * p_dist_mean + \
                             gamma * (beta ** 2) * m_dist_sup * m_dist_mean + \
                             alpha * beta * p_dist_sup * m_dist_mean + alpha * beta * p_dist_mean * m_dist_sup)

                    if bound > bound_star:
                        bound_star = bound
                        alpha_star = alpha
                        beta_star = beta
                        target_star = target

            # policy and model update
            if alpha_star > 0:
                policy = self.policy_combination(alpha_star, target_star, policy)
            if beta_star > 0:
                model = self.model_combination(beta_star, target_index, model)

            # performance evaluation
            Q = self.policy_q(policy, thresh)
            J_p_m = self.performance(policy, Q)

            self._utility_trace(J_p_m, alpha_star, beta_star, p_er_adv, m_er_adv,
                                p_dist_sup, p_dist_mean, m_dist_sup, m_dist_mean,
                                target_star, target_old, target_index, target_index_old, convergence, model)

            # policy chooser
            target_old = target_star
            d_mu = self.discounted_s_distribution(policy)
            p_er_adv, p_dist_sup, p_dist_mean, target = self.policy_chooser(policy, d_mu, Q)

            # model chooser
            target_index_old = target_index
            U = self.model_u(policy, thresh, Q)
            delta_mu = self.discounted_sa_distribution(policy, d_mu)
            m_er_adv, m_dist_sup, m_dist_mean, target_index = self.model_chooser(model, delta_mu, U)

        return policy, model

    # implementation of safe policy (and) model iteration
    # version which executes a full spi and then a smi
    def safe_policy_model_iteration_sequential(self, initial_policy, initial_model):

        # initializations
        gamma = self.gamma
        eps = self.eps
        thresh = self.threshold
        iteration_horizon = self.iteration_horizon
        self._reset_trace()


        # policy chooser
        policy = initial_policy
        Q = self.policy_q(policy, thresh)
        d_mu = self.discounted_s_distribution(policy)
        p_er_adv, p_dist_sup, p_dist_mean, target = self.policy_chooser(policy, d_mu, Q)
        target_old = target


        # check convergence condition
        convergence = eps / (1 - gamma)
        while p_er_adv > convergence and self.iteration < iteration_horizon:

            target_to_test = [(target, p_er_adv, p_dist_sup, p_dist_mean)]
            if self.use_target_trick:
                if not self.policy_equiv_check(target, target_old) and not self.policy_equiv_check(policy, target_old):
                    er_adv_old, dist_sup_old, dist_mean_old = self.policy_chooser_old(policy, target_old, d_mu, Q)
                    target_to_test.append((target_old, er_adv_old, dist_sup_old, dist_mean_old))

            alpha_star = 0.
            bound_star = 0.
            target_star = target

            for target, p_er_adv, p_dist_sup, p_dist_mean in target_to_test:

                alpha = ((1 - gamma) * p_er_adv) / (self.delta_q * gamma * \
                            p_dist_sup * p_dist_mean)

                alpha = np.clip(alpha, 0., 1.)

                bound = alpha * p_er_adv -\
                        (gamma / (1 - gamma) * self.delta_q / 2) * \
                        ((alpha ** 2) * p_dist_sup * p_dist_mean)

                if bound > bound_star:
                    bound_star = bound
                    alpha_star = alpha
                    target_star = target

            # policy and model update
            if alpha_star > 0:
                policy = self.policy_combination(alpha_star, target_star, policy)

            # performance evaluation
            Q = self.policy_q(policy, thresh)
            J_p_m = self.performance(policy, Q)

            self._utility_trace_p(J_p_m, alpha, p_er_adv, p_dist_sup,
                                  p_dist_mean, target, target_old, convergence)

            # policy chooser
            target_old = target_star
            d_mu = self.discounted_s_distribution(policy)
            p_er_adv, p_dist_sup, p_dist_mean, target = self.policy_chooser(policy, d_mu, Q)


        # model chooser
        model = initial_model
        U = self.model_u(policy, thresh, Q)
        delta_mu = self.discounted_sa_distribution(policy, d_mu)
        m_er_adv, m_dist_sup, m_dist_mean, target_index = self.model_chooser(model, delta_mu, U)
        target_index_old = target_index

        while m_er_adv > convergence and self.iteration < iteration_horizon:

            beta = ((1 - gamma) * m_er_adv) / (self.delta_q * (gamma ** 2) \
                        * m_dist_sup * m_dist_mean)

            beta = np.clip(beta, 0., 1.)

            bound =  beta * m_er_adv - \
                    (gamma / (1 - gamma) * self.delta_q / 2) * \
                    (gamma * (beta ** 2) * m_dist_sup * m_dist_mean)

            beta_star = beta
            bound_star = bound

            if beta_star > 0:
                model = self.model_combination(beta_star, target_index, model)

            # performance evaluation
            Q = self.policy_q(policy, thresh)
            J_p_m = self.performance(policy, Q)

            self._utility_trace_m(J_p_m, beta, m_er_adv, m_dist_sup,
                                  m_dist_mean,
                                  target_index, target_index_old, convergence,
                                  model)

            # model chooser
            target_index_old = target_index
            U = self.model_u(policy, thresh)
            delta_mu = self.discounted_sa_distribution(policy)
            m_er_adv, m_dist_sup, m_dist_mean, target_index = self.model_chooser(model, delta_mu, U)

        return policy, model

    # implementation of safe policy (and) model iteration
    # version which executes a single spi sweep and a smi sweep iteratively
    def safe_policy_model_alternated(self, initial_policy, initial_model):

        # initializations
        gamma = self.gamma
        eps = self.eps
        thresh = self.threshold
        iteration_horizon = self.iteration_horizon
        self._reset_trace()


        # policy chooser
        policy = initial_policy
        Q = self.policy_q(policy, thresh)
        d_mu = self.discounted_s_distribution(policy)
        p_er_adv, p_dist_sup, p_dist_mean, target = self.policy_chooser(policy, d_mu, Q)
        target_old = target

        # model chooser
        model = initial_model
        U = self.model_u(policy, thresh, Q)
        delta_mu = self.discounted_sa_distribution(policy, d_mu)
        m_er_adv, m_dist_sup, m_dist_mean, target_index = self.model_chooser(model, delta_mu, U)
        target_index_old = target_index


        # check convergence condition
        convergence = eps / (1 - gamma)
        while ((p_er_adv + m_er_adv) > convergence) and self.iteration < iteration_horizon:

            target_to_test = [(target, p_er_adv, p_dist_sup, p_dist_mean)]
            if self.use_target_trick:
                if not self.policy_equiv_check(target, target_old) and not self.policy_equiv_check(policy, target_old):
                    er_adv_old, dist_sup_old, dist_mean_old = self.policy_chooser_old(policy, target_old, d_mu, Q)
                    target_to_test.append((target_old, er_adv_old, dist_sup_old, dist_mean_old))

            bound_star = 0.
            alpha_star = 0.
            beta_star = 0.
            target_star = target

            for target, p_er_adv, p_dist_sup, p_dist_mean in target_to_test:

                alpha0 = ((1 - gamma) * p_er_adv) / (self.delta_q * gamma * \
                            p_dist_sup * p_dist_mean)
                beta0 = ((1 - gamma) * m_er_adv) / (self.delta_q * (gamma ** 2) \
                            * m_dist_sup * m_dist_mean)

                alpha0 = np.clip(alpha0, 0., 1.)
                beta0 = np.clip(beta0, 0., 1.)

                alt = [(alpha0, 0.), (0., beta0)]

                alpha, beta = alt[self.iteration % 2]

                bound = alpha * p_er_adv + beta * m_er_adv - \
                        (gamma / (1 - gamma) * self.delta_q / 2) * \
                        ((alpha ** 2) * p_dist_sup * p_dist_mean + \
                         gamma * (beta ** 2) * m_dist_sup * m_dist_mean + \
                         alpha * beta * p_dist_sup * m_dist_mean + alpha * beta * p_dist_mean * m_dist_sup)

                if bound > bound_star:
                    bound_star = bound
                    alpha_star = alpha
                    beta_star = beta
                    target_star = target

            # policy and model update
            if alpha_star > 0:
                policy = self.policy_combination(alpha_star, target_star, policy)
            if beta_star > 0:
                model = self.model_combination(beta_star, target_index, model)

            # performance evaluation
            Q = self.policy_q(policy, thresh)
            J_p_m = self.performance(policy, Q)

            self._utility_trace(J_p_m, alpha_star, beta_star, p_er_adv, m_er_adv,
                                p_dist_sup, p_dist_mean, m_dist_sup, m_dist_mean,
                                target_star, target_old, target_index, target_index_old, convergence, model)

            # policy chooser
            target_old = target_star
            d_mu = self.discounted_s_distribution(policy)
            p_er_adv, p_dist_sup, p_dist_mean, target = self.policy_chooser(policy, d_mu, Q)

            # model chooser
            target_index_old = target_index
            U = self.model_u(policy, thresh, Q)
            delta_mu = self.discounted_sa_distribution(policy, d_mu)
            m_er_adv, m_dist_sup, m_dist_mean, target_index = self.model_chooser(model, delta_mu, U)

        return policy, model

    # method to exactly evaluate a policy in terms of q-function
    # it returns Q: dictionary over states indexing array of values
    def policy_q(self, policy, threshold):

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


    # method to exactly compute the d_mu_P
    def discounted_s_distribution(self, policy):

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
            if a == nA:
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
            #d_mu_next = (1 - gamma) * mu + gamma * dot
            d_mu_next = mu + gamma * dot
            d_mu = d_mu_next
            count = count + 1

        d_mu = d_mu * (1 - gamma) / (1 - gamma ** h)

        assert(np.isclose(np.sum(d_mu), 1.))

        return d_mu


    # method which returns the estimated greedy policy
    # and the corresponding expected relative advantage
    def policy_chooser(self, policy, d_mu_pi, Q):

        # GREEDY POLICY COMPUTATION
        target_policy_rep = self.greedy_policy(Q)
        # instantiation of a target policy object
        target_policy = SpiPolicy(target_policy_rep)

        # EXPECTED RELATIVE ADVANTAGE COMPUTATION
        er_advantage = self.er_advantage(target_policy, policy, Q, d_mu_pi)

        # POLICY DISTANCE COMPUTATIONS
        distance_sup = self.pol_infinite_norm(target_policy, policy)
        distance_mean = self.pol_mean_distance(target_policy, policy, d_mu_pi)

        return er_advantage, distance_sup, distance_mean, target_policy


    # method mirroring pol_chooser with a fixed target as input
    def policy_chooser_old(self, policy, target, d_mu_pi, Q):

        # EXPECTED RELATIVE ADVANTAGE COMPUTATION
        er_advantage = self.er_advantage(target, policy, Q, d_mu_pi)

        # POLICY DISTANCE COMPUTATIONS
        distance_sup = self.pol_infinite_norm(target, policy)
        distance_mean = self.pol_mean_distance(target, policy, d_mu_pi)

        return er_advantage, distance_sup, distance_mean


    # method to compute the greedy policy given the q-function
    def greedy_policy(self, Q, tol=1e-3):

        # initializations
        mdp = self.mdp
        nS = mdp.nS
        nA = mdp.nA

        greedy_policy_rep = {s: [] for s in range(nS)}
        # loop to give maximum probability to the greedy action,
        # if more than one is greedy then uniform on the greedy actions
        for s in range(nS):
            q_array = Q[s]
            probabilities = np.zeros(nA)

            # uniform if more than one greedy
            max = np.max(q_array)
            # a = np.argwhere(q_array == max).flatten()
            a = np.argwhere(np.abs(q_array - max) < tol).flatten()
            probabilities[a] = 1. / len(a)
            greedy_policy_rep[s] = probabilities

            # # lexicograph order if more than one greedy
            # a = np.argmax(q_array)
            # probabilities[a] = 1
            # greedy_policy_rep[s] = probabilities

        return greedy_policy_rep


    # method to exactly compute the expected relative advantage
    def er_advantage(self, target, policy, Q, d_mu_pi):

        # initializations
        mdp = self.mdp
        nS = mdp.nS
        target_rep = target.get_rep()
        policy_rep = policy.get_rep()

        # relative advantage as array of states
        A = np.zeros(shape=nS)

        # loop to compute the relative advantage for each state
        for s in range(nS):
            pi_t_arr = target_rep[s]
            pi_arr = policy_rep[s]
            q_arr = Q[s]
            A[s] = np.dot(pi_t_arr - pi_arr, q_arr)

        # computation of the expected relative advantage
        er_advantage = np.dot(d_mu_pi, A)

        return er_advantage


    # method to compute the infinite norm between two given policies
    def pol_infinite_norm(self, policy1, policy2):

        # initializations
        policy1_rep = policy1.get_rep()
        policy2_rep = policy2.get_rep()
        mdp = self.mdp
        nS = mdp.nS
        nA = mdp.nA

        # infinite norm computation
        sum_s = np.zeros(nS)
        for s in range(nS):
            sum_a = 0
            for a in range(nA):
                diff_a = policy1_rep[s][a] - policy2_rep[s][a]
                sum_a = sum_a + np.abs(diff_a)
            sum_s[s] = sum_a
        max = np.max(sum_s)

        return max


    # method to compute the mean value distance between two given policies
    def pol_mean_distance(self, policy1, policy2, d_mu_pi):

        # initializations
        policy1_rep = policy1.get_rep()
        policy2_rep = policy2.get_rep()
        mdp = self.mdp
        nA = mdp.nA
        nS = mdp.nS

        # mean value computation
        sum_s = np.zeros(nS)
        for s in range(nS):
            sum_a = 0
            for a in range(nA):
                diff_a = policy1_rep[s][a] - policy2_rep[s][a]
                sum_a = sum_a + np.abs(diff_a)
            sum_s[s] = sum_a
        mean = np.dot(d_mu_pi, sum_s)

        return mean


    # method to check the equivalence fo two given policies
    def policy_equiv_check(self, policy1, policy2):

        policy1_rep = policy1.get_rep()
        policy2_rep = policy2.get_rep()
        outcome = True
        for s in range(self.mdp.nS):
            check = np.array_equiv(policy1_rep[s], policy2_rep[s])
            if not check:
                outcome = False
                break

        return outcome

    # method to combine linearly target and current with coefficient alfa
    def policy_combination(self, alfa, target, current):

        # initializations
        mdp = self.mdp
        nS = mdp.nS
        target_rep = target.get_rep()
        current_rep = current.get_rep()
        new_policy_rep = {s: [] for s in range(nS)}

        # linear combination
        for s in range(nS):
            new_policy_rep[s] = (alfa * target_rep[s]) + ((1 - alfa) * current_rep[s])

        # instantiation of the new policy as SpiPolicy
        new_policy = SpiPolicy(new_policy_rep)

        return new_policy


    # method to compute the U-function
    def model_u(self, policy, thresh, Q=None):

        # initializations
        mdp = self.mdp
        nS = mdp.nS
        nA = mdp.nA
        gamma = mdp.gamma
        P_sa = mdp.P_sa
        R = mdp.R
        policy_rep = policy.get_rep()

        # Q-function computation
        if Q is None:   #for efficiency reasons you can avoid recompunting Q
            Q = self.policy_q(policy, thresh)

        # U-function instantiation as (SA*S) matrix
        U = np.zeros(shape=(nS * nA, nS))

        # loop to fill the values in U
        for s1 in range(nS):
            q_arr = Q[s1]
            pi_arr = policy_rep[s1]
            V_s1 = np.dot(q_arr, pi_arr)
            U[np.nonzero(P_sa[:, s1]), s1] = R[s1] + gamma * V_s1
        return U


    # method to compute the discounted state,action distribution
    def discounted_sa_distribution(self, policy, d_mu=None):

        # initializations
        policy_rep = policy.get_rep()
        mdp = self.mdp
        nS = mdp.nS
        nA = mdp.nA

        # d_mu computation
        if d_mu is None:  #for efficiency reasons you can avoid recompunting d_mu
            d_mu = self.discounted_s_distribution(policy)

        # instantiation of delta_mu as SA vector
        delta_mu = np.zeros(nS * nA)

        # loop to fill the value in delta_mu
        a = 0
        s = 0
        for sa in range(nS * nA):
            if a == nA:
                a = 0
                s = s + 1
            delta_mu[sa] = d_mu[s] * policy_rep[s][a]
            a = a + 1

        assert(np.isclose(np.sum(delta_mu), 1.))

        return delta_mu


    # method which returns the target model,
    # er advantage, mean and sup distance
    def model_chooser(self, w, delta_mu, U):

        # initializations
        mdp = self.mdp
        PHI = mdp.get_phi()
        P_sa = mdp.P_sa

        # compute er_advantage for each model
        vertex_adv = self.vertex_advantage(delta_mu, U)

        # select the target model and target adv
        target_index = np.argmax(vertex_adv)
        er_advantage = vertex_adv[target_index]

        # compute the distances
        P_target = PHI[target_index]
        distance_sup = self.model_infinite_norm(P_target, P_sa)
        distance_mean = self.model_mean_distance(P_target, P_sa, delta_mu)

        return er_advantage, distance_sup, distance_mean, target_index


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
            sum_sa[sa] = np.sum(np.abs(diff_s))
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
            sum_sa[sa] = np.sum(np.abs(diff_s))
        mean = np.dot(delta_mu, sum_sa)

        return mean


    # method which compute the new combination
    # coefficients and calls the model_configuration()
    def model_combination(self, beta, target_index, w):

        # initializations
        mdp = self.mdp

        # coefficients computation
        w_next = w * (1 - beta)
        w_next[target_index] = beta + w[target_index] * (1 - beta)

        assert(np.isclose(np.sum(w_next), 1.))

        # model_configuration call
        mdp.model_configuration(w_next[0])

        return w_next


    # method to compute the performance of a given policy
    def performance(self, policy, Q):

        # initializations
        policy_rep = policy.get_rep()
        mdp = self.mdp
        mu = mdp.mu
        nS = mdp.nS

        # V-function computation
        V = np.zeros(nS)
        for s in range(nS):
            pi_arr = policy_rep[s]
            q_arr = Q[s]
            V[s] = np.dot(pi_arr, q_arr)

        # performance computation
        J_pi = np.dot(mu, V)

        return J_pi


    # utility method to print the algorithm trace
    # and to collect execution data
    def _utility_trace(self, J_p_m, alfa_star, beta_star, p_er_adv, m_er_adv,
                       p_dist_sup, p_dist_mean, m_dist_sup, m_dist_mean, target,
                       target_old, target_index, target_index_old, convergence, model):

        # data collections
        self.iterations.append(self.iteration)
        self.evaluations.append(J_p_m)
        self.alfas.append(alfa_star)
        self.betas.append(beta_star)
        self.p_advantages.append(p_er_adv)
        self.m_advantages.append(m_er_adv)
        self.p_dist_sup.append(p_dist_sup)
        self.p_dist_mean.append(p_dist_mean)
        self.m_dist_sup.append(m_dist_sup)
        self.m_dist_mean.append(m_dist_mean)
        self.coefficients.append(model[0])

        # target policy change check
        p_check_target = self.policy_equiv_check(target, target_old)

        # target model change check
        m_check_target = (target_index == target_index_old)

        # trace print
        print('----------------------')
        print('performance: {0}'.format(J_p_m))
        print('alfa/beta: {0}/{1}'.format(alfa_star, beta_star))
        print('iteration: {0}'.format(self.iteration))
        print('condition: {0}\n'.format(convergence))

        print('policy advantage: {0}'.format(p_er_adv))
        print('alfa star: {0}'.format(alfa_star))
        print('policy dist sup: {0}'.format(p_dist_sup))
        print('policy dist mean: {0}'.format(p_dist_mean))
        print('policy same target: {0}\n'.format(p_check_target))

        print('model advantage: {0}'.format(m_er_adv))
        print('beta star: {0}'.format(beta_star))
        print('model dist sup: {0}'.format(m_dist_sup))
        print('model dist mean: {0}'.format(m_dist_mean))
        print('model same target: {0}'.format(m_check_target))
        print('current model: {0}\n'.format(model))

        # iteration update
        self.iteration = self.iteration + 1


    # utility method to print the algorithm trace
    # and to collect execution data (SPI)
    def _utility_trace_p(self, J_p_m, alfa_star, p_er_adv, p_dist_sup,
                        p_dist_mean, target, target_old, convergence):

        # data collections
        self.iterations.append(self.iteration)
        self.evaluations.append(J_p_m)
        self.alfas.append(alfa_star)
        self.p_advantages.append(p_er_adv)
        self.p_dist_sup.append(p_dist_sup)
        self.p_dist_mean.append(p_dist_mean)

        # target policy change check
        p_check_target = self.policy_equiv_check(target, target_old)


        # trace print
        print('----------------------')
        print('performance: {0}'.format(J_p_m))
        print('iteration: {0}'.format(self.iteration))
        print('condition: {0}\n'.format(convergence))

        print('policy advantage: {0}'.format(p_er_adv))
        print('alfa star: {0}'.format(alfa_star))
        print('policy dist sup: {0}'.format(p_dist_sup))
        print('policy dist mean: {0}'.format(p_dist_mean))
        print('policy same target: {0}\n'.format(p_check_target))

        # iteration update
        self.iteration = self.iteration + 1


    # utility method to print the algorithm trace
    # and to collect execution data (SMI)
    def _utility_trace_m(self, J_p_m, beta_star, m_er_adv, m_dist_sup, m_dist_mean,
                         target_index, target_index_old, convergence, model):

        # data collections
        self.iterations.append(self.iteration)
        self.evaluations.append(J_p_m)
        self.betas.append(beta_star)
        self.m_advantages.append(m_er_adv)
        self.m_dist_sup.append(m_dist_sup)
        self.m_dist_mean.append(m_dist_mean)
        self.coefficients.append(model[0])

        # target model change check
        m_check_target = (target_index == target_index_old)

        # trace print
        print('----------------------')
        print('performance: {0}'.format(J_p_m))
        print('iteration: {0}'.format(self.iteration))
        print('condition: {0}\n'.format(convergence))

        print('model advantage: {0}'.format(m_er_adv))
        print('beta star: {0}'.format(beta_star))
        print('model dist sup: {0}'.format(m_dist_sup))
        print('model dist mean: {0}'.format(m_dist_mean))
        print('model same target: {0}'.format(m_check_target))
        print('current model: {0}\n'.format(model))

        # iteration update
        self.iteration = self.iteration + 1


    # utility method to reset the algorithm trace
    def _reset_trace(self):

        self.count = 0
        self.iteration = 0
        self.iterations = list()
        self.evaluations = list()
        self.p_advantages = list()
        self.m_advantages = list()
        self.p_dist_sup = list()
        self.p_dist_mean = list()
        self.m_dist_sup = list()
        self.m_dist_mean = list()
        self.alfas = list()
        self.betas = list()
        self.coefficients = list()