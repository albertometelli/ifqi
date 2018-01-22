import numpy as np
from spmi.utils.tictoc import tic, toc
from spmi.utils import evaluator
from spmi.utils.tabular_operations import policy_mean_tv_distance, policy_sup_tv_distance, model_sup_tv_distance, model_mean_tv_distance
from spmi.algorithms.policy_chooser import *
from spmi.algorithms.model_chooser import *
from spmi.utils import evaluator
from spmi.utils.tabular import *
from spmi.utils.tabular_operations import policy_convex_combination, model_convex_combination
import copy

class SPMI(object):

    def __init__(self,
                 mdp,
                 eps,
                 policy_chooser=None,
                 model_chooser=None,
                 max_iter=10000,
                 delta_q=None,
                 use_target_trick=True):
        '''
        This class implements Safe Policy Model Iteration

        :param mdp: the Markov decision process object
        :param eps: threshold to be use to stop iterations
        :param policy_chooser: an object implementing the choice of the target policy
        :param model_chooser: an object implementing the choice of the target model
        :param max_iter: maximum number of iterations to be performed
        :param delta_q: the value of DeltaQ, if None 1/(1-gamma) is used
        :param use_target_trick: whether to keep the current target if better
        '''

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

        if policy_chooser is None:
            self.policy_chooser = GreedyPolicyChooser(mdp.nS, mdp.nA)
        else:
            self.policy_chooser = policy_chooser

        if model_chooser is None:
            self.model_chooser = GreedyModelChooser(mdp.nS, mdp.nA)
        else:
            self.model_chooser = model_chooser

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
        self.p_change = list()
        self.m_change = list()


    # implementation of safe policy (and) model iteration
    def safe_policy_model_iteration(self, initial_policy, initial_model):

        # initializations
        gamma = self.gamma
        reward = TabularReward(self.mdp.P, self.mdp.nS, self.mdp.nA)
        mu = self.mdp.mu
        nS, nA = self.mdp.nS, self.mdp.nA
        horizon = self.horizon
        eps = self.eps
        iteration_horizon = self.iteration_horizon
        self._reset_trace()

        policy = initial_policy
        model = initial_model

        # policy chooser
        Q = evaluator.compute_q_function(policy, model, reward, gamma, horizon=horizon)
        d_mu = evaluator.compute_discounted_s_distribution(mu, \
                    policy, model, gamma, horizon, nS, nA)
        p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)
        target_policy_old = target_policy

        # model chooser
        U = evaluator.compute_u_function(policy, model, reward, gamma, horizon=horizon)
        delta_mu = evaluator.compute_discounted_sa_distribution(mu, \
                policy, model, gamma, horizon, nS, nA, d_mu)
        m_er_adv, m_dist_sup, m_dist_mean, target_model = self.model_chooser.choose(model, delta_mu, U)
        target_model_old = target_model

        # check convergence condition
        convergence = eps / (1 - gamma)
        #while (p_er_adv > convergence or m_er_adv > convergence) and self.iteration < iteration_horizon:
        while (p_er_adv > convergence or m_er_adv > convergence) and self.iteration < iteration_horizon:

            target_policies = [(target_policy, p_er_adv, p_dist_sup, p_dist_mean)]
            if self.use_target_trick:
                if not self.policy_equiv_check(target_policy, target_policy_old) and not self.policy_equiv_check(policy, target_policy_old):
                    er_adv_old = evaluator.compute_policy_er_advantage(target_policy_old, policy, Q, d_mu)
                    dist_sup_old = policy_sup_tv_distance(target_policy_old, policy)
                    dist_mean_old = policy_mean_tv_distance(target_policy_old, policy, d_mu)
                    target_policies.append((target_policy_old, er_adv_old, dist_sup_old, dist_mean_old))

            target_models = [(target_model, m_er_adv, m_dist_sup, m_dist_mean)]
            if self.use_target_trick:
                if not self.model_equiv_check(target_model, target_model_old) and not self.model_equiv_check(model, target_model_old):
                    er_adv_old = evaluator.compute_model_er_advantage(target_model_old, model, U, delta_mu)
                    dist_sup_old = policy_sup_tv_distance(target_model_old, model)
                    dist_mean_old = policy_mean_tv_distance(target_model_old, model, delta_mu)
                    target_models.append((target_model_old, er_adv_old, dist_sup_old, dist_mean_old))

            bound_star = 0.
            alpha_star = 0.
            beta_star = 0.
            target_policy_star = target_policy
            target_model_star = target_model
            p_er_adv_star, p_dist_sup_star, p_dist_mean_star = None, None, None
            m_er_adv_star, m_dist_sup_star, m_dist_mean_star = None, None, None


            for target_policy, p_er_adv, p_dist_sup, p_dist_mean in target_policies:
                for target_model, m_er_adv, m_dist_sup, m_dist_mean in target_models:

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
                            target_policy_star = target_policy
                            target_model_star = target_model
                            p_er_adv_star, p_dist_sup_star, p_dist_mean_star = p_er_adv, p_dist_sup, p_dist_mean
                            m_er_adv_star, m_dist_sup_star, m_dist_mean_star = m_er_adv, m_dist_sup, m_dist_mean

            # policy and model update
            if alpha_star > 0:
                policy = self.policy_combination(alpha_star, target_policy_star, policy)
            if beta_star > 0:
                model = self.model_combination(beta_star, target_model_star, model)

            # performance evaluation
            Q = evaluator.compute_q_function(policy, model, reward, gamma, horizon=horizon)
            J_p_m = evaluator.compute_performance(mu, reward, \
                    policy, model, gamma, horizon, nS, nA)

            self._utility_trace(J_p_m, alpha_star, beta_star, p_er_adv_star, m_er_adv_star,
                                p_dist_sup_star, p_dist_mean_star, m_dist_sup_star, m_dist_mean_star,
                                target_policy_star, target_policy_old, target_model_star, target_model_old,
                                 convergence)

            mu = self.mdp.mu

            # policy chooser
            target_policy_old = target_policy_star
            d_mu = evaluator.compute_discounted_s_distribution(mu, \
                policy, model, gamma, horizon, nS, nA)
            p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)

            # model chooser
            target_model_old = target_model_star
            U = evaluator.compute_u_function(policy, model, reward, gamma, horizon=horizon)
            delta_mu = evaluator.compute_discounted_sa_distribution(mu, \
                policy, model, gamma, horizon, nS, nA, d_mu)
            m_er_adv, m_dist_sup, m_dist_mean, target_model = self.model_chooser.choose(model, delta_mu, U)

        return policy, model

    # implementation of safe policy (and) model iteration
    # version with sup distances in alfa, beta computation
    def safe_policy_model_iteration_sup(self, initial_policy, initial_model):

        # initializations
        gamma = self.gamma
        reward = TabularReward(self.mdp.P, self.mdp.nS, self.mdp.nA)
        mu = self.mdp.mu
        nS, nA = self.mdp.nS, self.mdp.nA
        horizon = self.horizon
        eps = self.eps
        iteration_horizon = self.iteration_horizon
        self._reset_trace()

        policy = initial_policy
        model = initial_model

        # policy chooser
        Q = evaluator.compute_q_function(policy, model, reward, gamma, horizon=horizon)
        d_mu = evaluator.compute_discounted_s_distribution(mu, \
                    policy, model, gamma, horizon, nS, nA)
        p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)
        target_policy_old = target_policy

        # model chooser
        U = evaluator.compute_u_function(policy, model, reward, gamma, horizon=horizon)
        delta_mu = evaluator.compute_discounted_sa_distribution(mu, \
                policy, model, gamma, horizon, nS, nA, d_mu)
        m_er_adv, m_dist_sup, m_dist_mean, target_model = self.model_chooser.choose(model, delta_mu, U)
        target_model_old = target_model

        # check convergence condition
        convergence = eps / (1 - gamma)
        while ((p_er_adv + m_er_adv) > convergence) and self.iteration < iteration_horizon:

            target_policies = [(target_policy, p_er_adv, p_dist_sup, p_dist_mean)]
            if self.use_target_trick:
                if not self.policy_equiv_check(target_policy, target_policy_old) and not self.policy_equiv_check(policy, target_policy_old):
                    er_adv_old = evaluator.compute_policy_er_advantage(target_policy_old, policy, Q, d_mu)
                    dist_sup_old = policy_sup_tv_distance(target_policy_old, policy)
                    dist_mean_old = policy_mean_tv_distance(target_policy_old, policy, d_mu)
                    target_policies.append((target_policy_old, er_adv_old, dist_sup_old, dist_mean_old))

            target_models = [(target_model, m_er_adv, m_dist_sup, m_dist_mean)]
            if self.use_target_trick:
                if not self.model_equiv_check(target_model, target_model_old) and not self.model_equiv_check(model, target_model_old):
                    er_adv_old = evaluator.compute_model_er_advantage(target_model_old, model, U, delta_mu)
                    dist_sup_old = policy_sup_tv_distance(target_model_old, model)
                    dist_mean_old = policy_mean_tv_distance(target_model_old, model, delta_mu)
                    target_models.append((target_model_old, er_adv_old, dist_sup_old, dist_mean_old))

            bound_star = 0.
            alpha_star = 0.
            beta_star = 0.
            target_policy_star = target_policy
            target_model_star = target_model
            p_er_adv_star, p_dist_sup_star, p_dist_mean_star = None, None, None
            m_er_adv_star, m_dist_sup_star, m_dist_mean_star = None, None, None


            for target_policy, p_er_adv, p_dist_sup, p_dist_mean in target_policies:
                for target_model, m_er_adv, m_dist_sup, m_dist_mean in target_models:

                    alpha0 = ((1 - gamma) * p_er_adv) / (self.delta_q * gamma * \
                                p_dist_sup * p_dist_sup + 1e-24)
                    alpha1 = ((1 - gamma) * p_er_adv) / (self.delta_q * gamma * \
                                p_dist_sup * p_dist_sup + 1e-24) - \
                                m_dist_sup / (p_dist_sup + 1e-24)
                    beta0 = ((1 - gamma) * m_er_adv) / (self.delta_q * (gamma ** 2) \
                                * m_dist_sup * m_dist_sup)
                    beta1 = ((1 - gamma) * m_er_adv) / (self.delta_q * (gamma ** 2) \
                                * m_dist_sup * m_dist_sup) - 1. / gamma * \
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
                            target_policy_star = target_policy
                            target_model_star = target_model
                            p_er_adv_star, p_dist_sup_star, p_dist_mean_star = p_er_adv, p_dist_sup, p_dist_mean
                            m_er_adv_star, m_dist_sup_star, m_dist_mean_star = m_er_adv, m_dist_sup, m_dist_mean

            # policy and model update
            if alpha_star > 0:
                policy = self.policy_combination(alpha_star, target_policy_star, policy)
            if beta_star > 0:
                model = self.model_combination(beta_star, target_model_star, model)

            # performance evaluation
            Q = evaluator.compute_q_function(policy, model, reward, gamma, horizon=horizon)
            J_p_m = evaluator.compute_performance(mu, reward, \
                    policy, model, gamma, horizon, nS, nA)

            self._utility_trace(J_p_m, alpha_star, beta_star, p_er_adv_star, m_er_adv_star,
                                p_dist_sup_star, p_dist_mean_star, m_dist_sup_star, m_dist_mean_star,
                                target_policy_star, target_policy_old, target_model_star, target_model_old,
                                 convergence)

            # policy chooser
            target_policy_old = target_policy_star
            d_mu = evaluator.compute_discounted_s_distribution(mu, \
                policy, model, gamma, horizon, nS, nA)
            p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)

            # model chooser
            target_model_old = target_model_star
            U = evaluator.compute_u_function(policy, model, reward, gamma, horizon=horizon)
            delta_mu = evaluator.compute_discounted_sa_distribution(mu, \
                policy, model, gamma, horizon, nS, nA, d_mu)
            m_er_adv, m_dist_sup, m_dist_mean, target_model = self.model_chooser.choose(model, delta_mu, U)

        return policy, model

    # implementation of safe policy (and) model iteration
    # version which avoids full step to the target policy (model)
    def safe_policy_model_iteration_no_full_step(self, initial_policy, initial_model):

        # initializations
        gamma = self.gamma
        reward = TabularReward(self.mdp.P, self.mdp.nS, self.mdp.nA)
        mu = self.mdp.mu
        nS, nA = self.mdp.nS, self.mdp.nA
        horizon = self.horizon
        eps = self.eps
        iteration_horizon = self.iteration_horizon
        self._reset_trace()

        policy = initial_policy
        model = initial_model

        # policy chooser
        Q = evaluator.compute_q_function(policy, model, reward, gamma, horizon=horizon)
        d_mu = evaluator.compute_discounted_s_distribution(mu, \
                    policy, model, gamma, horizon, nS, nA)
        p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)
        target_policy_old = target_policy

        # model chooser
        U = evaluator.compute_u_function(policy, model, reward, gamma, horizon=horizon)
        delta_mu = evaluator.compute_discounted_sa_distribution(mu, \
                 policy, model, gamma, horizon, nS, nA, d_mu)
        m_er_adv, m_dist_sup, m_dist_mean, target_model = self.model_chooser.choose(
            model, delta_mu, U)
        target_model_old = target_model

        # check convergence condition
        convergence = eps / (1 - gamma)
        while ((p_er_adv + m_er_adv) > convergence) and self.iteration < iteration_horizon:

            target_policies = [(target_policy, p_er_adv, p_dist_sup, p_dist_mean)]
            if self.use_target_trick:
                if not self.policy_equiv_check(target_policy, target_policy_old) and not self.policy_equiv_check(policy, target_policy_old):
                    er_adv_old = evaluator.compute_policy_er_advantage(target_policy_old, policy, Q, d_mu)
                    dist_sup_old = policy_sup_tv_distance(target_policy_old, policy)
                    dist_mean_old = policy_mean_tv_distance(target_policy_old, policy, d_mu)
                    target_policies.append((target_policy_old, er_adv_old, dist_sup_old, dist_mean_old))

            target_models = [(target_model, m_er_adv, m_dist_sup, m_dist_mean)]
            if self.use_target_trick:
                if not self.model_equiv_check(target_model, target_model_old) and not self.model_equiv_check(model, target_model_old):
                    er_adv_old = evaluator.compute_model_er_advantage(target_model_old, model, U, delta_mu)
                    dist_sup_old = policy_sup_tv_distance(target_model_old, model)
                    dist_mean_old = policy_mean_tv_distance(target_model_old, model, delta_mu)
                    target_models.append((target_model_old, er_adv_old, dist_sup_old, dist_mean_old))

            bound_star = 0.
            alpha_star = 0.
            beta_star = 0.
            target_policy_star = target_policy
            target_model_star = target_model
            p_er_adv_star, p_dist_sup_star, p_dist_mean_star = None, None, None
            m_er_adv_star, m_dist_sup_star, m_dist_mean_star = None, None, None


            for target_policy, p_er_adv, p_dist_sup, p_dist_mean in target_policies:
                for target_model, m_er_adv, m_dist_sup, m_dist_mean in target_models:

                    alpha0 = ((1 - gamma) * p_er_adv) / (self.delta_q * gamma * \
                                p_dist_sup * p_dist_mean + 1e-24)
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
                            target_policy_star = target_policy
                            target_model_star = target_model
                            p_er_adv_star, p_dist_sup_star, p_dist_mean_star = p_er_adv, p_dist_sup, p_dist_mean
                            m_er_adv_star, m_dist_sup_star, m_dist_mean_star = m_er_adv, m_dist_sup, m_dist_mean

            # policy and model update
            if alpha_star > 0:
                policy = self.policy_combination(alpha_star, target_policy_star, policy)
            if beta_star > 0:
                model = self.model_combination(beta_star, target_model_star, model)

            # performance evaluation
            Q = evaluator.compute_q_function(policy, model, reward, gamma, horizon=horizon)
            J_p_m = evaluator.compute_performance(mu, reward, \
                    policy, model, gamma, horizon, nS, nA)

            self._utility_trace(J_p_m, alpha_star, beta_star, p_er_adv_star, m_er_adv_star,
                                p_dist_sup_star, p_dist_mean_star, m_dist_sup_star, m_dist_mean_star,
                                target_policy_star, target_policy_old, target_model_star, target_model_old,
                                 convergence)

            # policy chooser
            target_policy_old = target_policy_star
            d_mu = evaluator.compute_discounted_s_distribution(mu, \
                policy, model, gamma, horizon, nS, nA)
            p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)

            # model chooser
            target_model_old = target_model_star
            U = evaluator.compute_u_function(policy, model, reward, gamma, horizon=horizon)
            delta_mu = evaluator.compute_discounted_sa_distribution(mu, \
                policy, model, gamma, horizon, nS, nA, d_mu)
            m_er_adv, m_dist_sup, m_dist_mean, target_model = self.model_chooser.choose(model, delta_mu, U)

        return policy, model

    # implementation of safe policy (and) model iteration
    # version which executes a full spi and then a smi
    def safe_policy_model_iteration_sequential(self, initial_policy, initial_model):

        # initializations
        gamma = self.gamma
        reward = TabularReward(self.mdp.P, self.mdp.nS, self.mdp.nA)
        mu = self.mdp.mu
        nS, nA = self.mdp.nS, self.mdp.nA
        horizon = self.horizon
        eps = self.eps
        iteration_horizon = self.iteration_horizon
        self._reset_trace()

        policy = initial_policy
        model = initial_model

        # policy chooser
        Q = evaluator.compute_q_function(policy, model, reward, gamma, horizon=horizon)
        d_mu = evaluator.compute_discounted_s_distribution(mu, \
                    policy, model, gamma, horizon, nS, nA)
        p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)
        target_policy_old = target_policy

        # check convergence condition
        convergence = eps / (1 - gamma)
        while p_er_adv > convergence and self.iteration < iteration_horizon:

            target_policies = [(target_policy, p_er_adv, p_dist_sup, p_dist_mean)]
            if self.use_target_trick:
                if not self.policy_equiv_check(target_policy, target_policy_old) and not self.policy_equiv_check(policy, target_policy_old):
                    er_adv_old = evaluator.compute_policy_er_advantage(target_policy_old, policy, Q, d_mu)
                    dist_sup_old = policy_sup_tv_distance(target_policy_old, policy)
                    dist_mean_old = policy_mean_tv_distance(target_policy_old, policy, d_mu)
                    target_policies.append((target_policy_old, er_adv_old, dist_sup_old, dist_mean_old))

            bound_star = 0.
            alpha_star = 0.

            target_policy_star = target_policy
            p_er_adv_star, p_dist_sup_star, p_dist_mean_star = None, None, None


            for target_policy, p_er_adv, p_dist_sup, p_dist_mean in target_policies:

                alpha = ((1 - gamma) * p_er_adv) / (self.delta_q * gamma * \
                            p_dist_sup * p_dist_mean + 1e-24)

                alpha = np.clip(alpha, 0., 1.)

                bound = alpha * p_er_adv - \
                        (gamma / (1 - gamma) * self.delta_q / 2) * \
                        ((alpha ** 2) * p_dist_sup * p_dist_mean)

                if bound > bound_star:
                    bound_star = bound
                    alpha_star = alpha
                    target_policy_star = target_policy
                    p_er_adv_star, p_dist_sup_star, p_dist_mean_star = p_er_adv, p_dist_sup, p_dist_mean

            # policy update
            if alpha_star > 0:
                policy = self.policy_combination(alpha_star, target_policy_star, policy)

            # performance evaluation
            Q = evaluator.compute_q_function(policy, model, reward, gamma, horizon=horizon)
            J_p_m = evaluator.compute_performance(mu, reward, \
                    policy, model, gamma, horizon, nS, nA)

            self._utility_trace_p(J_p_m, alpha_star, p_er_adv_star, p_dist_sup_star,
                                  p_dist_mean_star, target_policy, target_policy_old, convergence)

            # policy chooser
            target_policy_old = target_policy_star
            d_mu = evaluator.compute_discounted_s_distribution(mu, \
                policy, model, gamma, horizon, nS, nA)
            p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)


        # model chooser
        U = evaluator.compute_u_function(policy, model, reward, gamma, horizon=horizon)
        delta_mu = evaluator.compute_discounted_sa_distribution(mu, \
                      policy, model, gamma, horizon, nS, nA)
        m_er_adv, m_dist_sup, m_dist_mean, target_model = self.model_chooser.choose(model, delta_mu, U)
        target_model_old = target_model
        # check convergence condition
        convergence = eps / (1 - gamma)
        while m_er_adv > convergence and self.iteration < iteration_horizon:

            target_models = [(target_model, m_er_adv, m_dist_sup, m_dist_mean)]
            if self.use_target_trick:
                if not self.model_equiv_check(target_model, target_model_old) and not self.model_equiv_check(model, target_model_old):
                    er_adv_old = evaluator.compute_model_er_advantage(target_model_old, model, U, delta_mu)
                    dist_sup_old = policy_sup_tv_distance(target_model_old, model)
                    dist_mean_old = policy_mean_tv_distance(target_model_old, model, delta_mu)
                    target_models.append((target_model_old, er_adv_old, dist_sup_old, dist_mean_old))

            bound_star = 0.
            beta_star = 0.
            target_model_star = target_model
            m_er_adv_star, m_dist_sup_star, m_dist_mean_star = None, None, None


            for target_model, m_er_adv, m_dist_sup, m_dist_mean in target_models:

                beta = ((1 - gamma) * m_er_adv) / (self.delta_q * (gamma ** 2) \
                            * m_dist_sup * m_dist_mean)

                beta = np.clip(beta, 0., 1.)

                bound = beta * m_er_adv - \
                        (gamma / (1 - gamma) * self.delta_q / 2) * \
                         gamma * (beta ** 2) * m_dist_sup * m_dist_mean

                if bound > bound_star:
                    bound_star = bound
                    beta_star = beta
                    target_model_star = target_model
                    m_er_adv_star, m_dist_sup_star, m_dist_mean_star = m_er_adv, m_dist_sup, m_dist_mean

            # model update
            if beta_star > 0:
                model = self.model_combination(beta_star, target_model_star, model)

            # performance evaluation
            J_p_m = evaluator.compute_performance(mu, reward, \
                    policy, model, gamma, horizon, nS, nA)

            self._utility_trace_m(J_p_m, beta_star, m_er_adv_star, m_dist_sup_star, m_dist_mean_star,
                                 target_model_star, target_model_old, convergence)

            # model chooser
            target_model_old = target_model_star
            U = evaluator.compute_u_function(policy, model, reward, gamma, horizon=horizon)
            delta_mu = evaluator.compute_discounted_sa_distribution(mu, \
                policy, model, gamma, horizon, nS, nA)
            m_er_adv, m_dist_sup, m_dist_mean, target_model = self.model_chooser.choose(model, delta_mu, U)

        return policy, model

    # implementation of safe policy (and) model iteration
    # version which executes a single spi sweep and a smi sweep iteratively
    def safe_policy_model_iteration_alternated(self, initial_policy, initial_model):

        # initializations
        gamma = self.gamma
        reward = TabularReward(self.mdp.P, self.mdp.nS, self.mdp.nA)
        mu = self.mdp.mu
        nS, nA = self.mdp.nS, self.mdp.nA
        horizon = self.horizon
        eps = self.eps
        iteration_horizon = self.iteration_horizon
        self._reset_trace()

        policy = initial_policy
        model = initial_model

        # policy chooser
        Q = evaluator.compute_q_function(policy, model, reward, gamma, horizon=horizon)
        d_mu = evaluator.compute_discounted_s_distribution(mu, \
                    policy, model, gamma, horizon, nS, nA)
        p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)
        target_policy_old = target_policy

        # model chooser
        U = evaluator.compute_u_function(policy, model, reward, gamma, horizon=horizon)
        delta_mu = evaluator.compute_discounted_sa_distribution(mu, \
                policy, model, gamma, horizon, nS, nA, d_mu)
        m_er_adv, m_dist_sup, m_dist_mean, target_model = self.model_chooser.choose(
            model, delta_mu, U)
        target_model_old = target_model

        # check convergence condition
        convergence = eps / (1 - gamma)
        while ((p_er_adv + m_er_adv) > convergence) and self.iteration < iteration_horizon:

            target_policies = [(target_policy, p_er_adv, p_dist_sup, p_dist_mean)]
            if self.use_target_trick:
                if not self.policy_equiv_check(target_policy, target_policy_old) and not self.policy_equiv_check(policy, target_policy_old):
                    er_adv_old = evaluator.compute_policy_er_advantage(target_policy_old, policy, Q, d_mu)
                    dist_sup_old = policy_sup_tv_distance(target_policy_old, policy)
                    dist_mean_old = policy_mean_tv_distance(target_policy_old, policy, d_mu)
                    target_policies.append((target_policy_old, er_adv_old, dist_sup_old, dist_mean_old))

            target_models = [(target_model, m_er_adv, m_dist_sup, m_dist_mean)]
            if self.use_target_trick:
                if not self.model_equiv_check(target_model, target_model_old) and not self.model_equiv_check(model, target_model_old):
                    er_adv_old = evaluator.compute_model_er_advantage(target_model_old, model, U, delta_mu)
                    dist_sup_old = policy_sup_tv_distance(target_model_old, model)
                    dist_mean_old = policy_mean_tv_distance(target_model_old, model, delta_mu)
                    target_models.append((target_model_old, er_adv_old, dist_sup_old, dist_mean_old))

            bound_star = -1.
            alpha_star = 0.
            beta_star = 0.
            target_policy_star = target_policy
            target_model_star = target_model
            p_er_adv_star, p_dist_sup_star, p_dist_mean_star = None, None, None
            m_er_adv_star, m_dist_sup_star, m_dist_mean_star = None, None, None


            for target_policy, p_er_adv, p_dist_sup, p_dist_mean in target_policies:
                for target_model, m_er_adv, m_dist_sup, m_dist_mean in target_models:

                    alpha0 = ((1 - gamma) * p_er_adv) / (self.delta_q * gamma * \
                                p_dist_sup * p_dist_mean + 1e-24)
                    beta0 = ((1 - gamma) * m_er_adv) / (self.delta_q * (gamma ** 2) \
                                * m_dist_sup * m_dist_mean + 1e-24)

                    alpha0 = np.clip(alpha0, 0., 1.)
                    beta0 = np.clip(beta0, 0., 1.)

                    if self.iteration % 2 == 0:
                        li = [(alpha0, 0.)]
                    else:
                        li = [(0., beta0)]

                    for alpha, beta in li:

                        bound = alpha * p_er_adv + beta * m_er_adv - \
                                (gamma / (1 - gamma) * self.delta_q / 2) * \
                                ((alpha ** 2) * p_dist_sup * p_dist_mean + \
                                 gamma * (beta ** 2) * m_dist_sup * m_dist_mean + \
                                 alpha * beta * p_dist_sup * m_dist_mean + alpha * beta * p_dist_mean * m_dist_sup)

                        if bound > bound_star:
                            bound_star = bound
                            alpha_star = alpha
                            beta_star = beta
                            target_policy_star = target_policy
                            target_model_star = target_model
                            p_er_adv_star, p_dist_sup_star, p_dist_mean_star = p_er_adv, p_dist_sup, p_dist_mean
                            m_er_adv_star, m_dist_sup_star, m_dist_mean_star = m_er_adv, m_dist_sup, m_dist_mean

            # policy and model update
            if alpha_star > 0:
                policy = self.policy_combination(alpha_star, target_policy_star, policy)
            if beta_star > 0:
                model = self.model_combination(beta_star, target_model_star, model)

            # performance evaluation
            Q = evaluator.compute_q_function(policy, model, reward, gamma, horizon=horizon)
            J_p_m = evaluator.compute_performance(mu, reward, \
                    policy, model, gamma, horizon, nS, nA)

            self._utility_trace(J_p_m, alpha_star, beta_star, p_er_adv_star, m_er_adv_star,
                                p_dist_sup_star, p_dist_mean_star, m_dist_sup_star, m_dist_mean_star,
                                target_policy_star, target_policy_old, target_model_star, target_model_old,
                                 convergence)

            # policy chooser
            target_policy_old = target_policy_star
            d_mu = evaluator.compute_discounted_s_distribution(mu, \
                policy, model, gamma, horizon, nS, nA)
            p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)

            # model chooser
            target_model_old = target_model_star
            U = evaluator.compute_u_function(policy, model, reward, gamma, horizon=horizon)
            delta_mu = evaluator.compute_discounted_sa_distribution(mu, \
                policy, model, gamma, horizon, nS, nA, d_mu)
            m_er_adv, m_dist_sup, m_dist_mean, target_model = self.model_chooser.choose(model, delta_mu, U)

        return policy, model

    # method to check the equivalence fo two given policies
    def policy_equiv_check(self, policy1, policy2):

        policy1_matrix = policy1.get_matrix()
        policy2_matrix = policy2.get_matrix()

        return np.array_equal(policy1_matrix, policy2_matrix)

    # method to combine linearly target and current with coefficient alfa
    def policy_combination(self, alfa, target, current):
        new_policy =  policy_convex_combination(target, current, alfa)
        return new_policy

    # method to check the equivalence fo two given models
    def model_equiv_check(self, model1, model2):

        model1_matrix = model1.get_matrix()
        model2_matrix = model2.get_matrix()

        return np.array_equal(model1_matrix, model2_matrix)

    # method which compute the new combination
    # coefficients and calls the model_configuration()
    def model_combination(self, beta, target, current):
        new_model = model_convex_combination(self.mdp.P, target, current, beta)
        self.mdp.set_model(new_model.get_rep())
        self.mdp.P_sa = new_model.get_matrix()
        return new_model

    # utility method to print the algorithm trace
    # and to collect execution data
    def _utility_trace(self, J_p_m, alfa_star, beta_star, p_er_adv, m_er_adv,
                       p_dist_sup, p_dist_mean, m_dist_sup, m_dist_mean,
                       target_policy, target_policy_old, target_model,
                       target_model_old,convergence):

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

        # target change check
        p_check_target = self.policy_equiv_check(target_policy, target_policy_old)
        m_check_target = self.model_equiv_check(target_model, target_model_old)
        self.p_change.append(p_check_target)
        self.m_change.append(m_check_target)

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


        print('model advantage: {0}'.format(m_er_adv))
        print('beta star: {0}'.format(beta_star))
        print('model dist sup: {0}'.format(m_dist_sup))
        print('model dist mean: {0}'.format(m_dist_mean))

        # coefficient computation and print
        if isinstance(self.model_chooser, SetModelChooser) and len(self.model_chooser.model_set) == 2:
            P = self.mdp.P_sa
            P1 = self.model_chooser.model_set[0].get_matrix()
            P2 = self.model_chooser.model_set[1].get_matrix()
            k = (P - P2) / (P1 - P2 + 1e-24)
            k = np.max(k)
            print('\ncurrent k: {0}'.format(k))
            self.coefficients.append(k)

        # iteration update
        self.iteration = self.iteration + 1


    # utility method to print the algorithm trace
    # and to collect execution data (SPI)
    def _utility_trace_p(self, J_p_m, alfa_star, p_er_adv, p_dist_sup,
                        p_dist_mean, target_policy, target_policy_old, convergence):

        # data collections
        self.iterations.append(self.iteration)
        self.evaluations.append(J_p_m)
        self.alfas.append(alfa_star)
        self.p_advantages.append(p_er_adv)
        self.p_dist_sup.append(p_dist_sup)
        self.p_dist_mean.append(p_dist_mean)

        self.betas.append(0.)
        self.m_advantages.append(np.nan)
        self.m_dist_sup.append(np.nan)
        self.m_dist_mean.append(np.nan)


        # target change check
        p_check_target = self.policy_equiv_check(target_policy, target_policy_old)
        self.p_change.append(p_check_target)
        self.m_change.append(True)


        # trace print
        print('----------------------')
        print('performance: {0}'.format(J_p_m))
        print('iteration: {0}'.format(self.iteration))
        print('condition: {0}\n'.format(convergence))

        print('policy advantage: {0}'.format(p_er_adv))
        print('alfa star: {0}'.format(alfa_star))
        print('policy dist sup: {0}'.format(p_dist_sup))
        print('policy dist mean: {0}'.format(p_dist_mean))

        # coefficient computation and print
        if isinstance(self.model_chooser, SetModelChooser) and len(self.model_chooser.model_set) == 2:
            P = self.mdp.P_sa
            P1 = self.model_chooser.model_set[0].get_matrix()
            P2 = self.model_chooser.model_set[1].get_matrix()
            k = (P - P2) / (P1 - P2 + 1e-24)
            k = np.max(k)
            print('\ncurrent k: {0}'.format(k))
            self.coefficients.append(k)

        # iteration update
        self.iteration = self.iteration + 1


    # utility method to print the algorithm trace
    # and to collect execution data (SMI)
    def _utility_trace_m(self, J_p_m, beta_star, m_er_adv, m_dist_sup, m_dist_mean,
                         target_model, target_model_old, convergence):

        # data collections
        self.iterations.append(self.iteration)
        self.evaluations.append(J_p_m)
        self.betas.append(beta_star)
        self.m_advantages.append(m_er_adv)
        self.m_dist_sup.append(m_dist_sup)
        self.m_dist_mean.append(m_dist_mean)

        self.alfas.append(0.)
        self.p_advantages.append(np.nan)
        self.p_dist_sup.append(np.nan)
        self.p_dist_mean.append(np.nan)

        # target change check
        m_check_target = self.model_equiv_check(target_model, target_model_old)
        self.m_change.append(m_check_target)
        self.p_change.append(True)

        # trace print
        print('----------------------')
        print('performance: {0}'.format(J_p_m))
        print('iteration: {0}'.format(self.iteration))
        print('condition: {0}\n'.format(convergence))

        print('model advantage: {0}'.format(m_er_adv))
        print('beta star: {0}'.format(beta_star))
        print('model dist sup: {0}'.format(m_dist_sup))
        print('model dist mean: {0}'.format(m_dist_mean))

        # coefficient computation and print
        if isinstance(self.model_chooser, SetModelChooser) and len(self.model_chooser.model_set) == 2:
            P = self.mdp.P_sa
            P1 = self.model_chooser.model_set[0].get_matrix()
            P2 = self.model_chooser.model_set[1].get_matrix()
            k = (P - P2) / (P1 - P2 + 1e-24)
            k = np.max(k)
            print('\ncurrent k: {0}'.format(k))
            self.coefficients.append(k)

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
        self.p_change = list()
        self.m_change = list()

    # public method to save the execution data into
    # a csv file (directory path as parameter)
    def save_simulation(self, dir_path, file_name, entries=None):

        header_string = 'iterations;evaluations;p_advantages;m_advantages;' \
                        'p_dist_sup;p_dist_mean;m_dist_sup;m_dist_mean;alfa;beta;p_change;m_change'

        # if coefficients not empty we are using a parametric model
        execution_data = [self.iterations, self.evaluations,
                          self.p_advantages, self.m_advantages,
                          self.p_dist_sup, self.p_dist_mean,
                          self.m_dist_sup, self.m_dist_mean,
                          self.alfas, self.betas, self.p_change, self.m_change]

        if isinstance(self.model_chooser, SetModelChooser):

            header_string = header_string + ';coefficient'

            execution_data = [self.iterations, self.evaluations,
                              self.p_advantages, self.m_advantages,
                              self.p_dist_sup, self.p_dist_mean,
                              self.m_dist_sup, self.m_dist_mean,
                              self.alfas, self.betas, self.p_change,
                              self.m_change, self.coefficients]

        execution_data = np.array(execution_data).T

        if entries is not None:
            filter = np.arange(0, len(execution_data), len(execution_data) / entries)
            execution_data = execution_data[filter]


        np.savetxt(dir_path + '/' + file_name, execution_data,
                   delimiter=';', header=header_string, fmt='%.30e')