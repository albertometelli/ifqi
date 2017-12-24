import numpy as np

from spmi.utils.uniform_policy import UniformPolicy
from spmi.utils.spi_policy import SpiPolicy


class SPI(object):

    def __init__(self, mdp, eps, delta):
        """
        Safe Policy Iterator:
        object that enable the call for exact spi algorithm

        :param mdp: mdp to solve
        :param eps: accuracy of the spi
        :param delta: correct estimation probability
        """
        self.mdp = mdp
        self.gamma = mdp.gamma
        self.horizon = mdp.horizon
        self.eps = eps
        self.delta = delta
        self.uniform_policy = UniformPolicy(mdp)

        # attributes related to
        # trace print procedures
        self.count = 0
        self.iteration = 0
        self.iterations = list()
        self.evaluations = list()
        self.advantages = list()
        self.distances_sup = list()
        self.distances_mean = list()
        self.alfas = list()


    # implementation of whole algorithm: PolChooser + improvement
    # version with mean value, without the target trick
    def safe_policy_iteration(self, initial_policy):

        # initializations
        gamma = self.gamma
        eps = self.eps
        policy = initial_policy
        prev_target_policy = initial_policy
        d_mu_thresh = 0.0001
        Q_thresh = 0.0001
        iteration_horizon = 3000

        # update policy until convergence
        d_mu_pi = self.discounted_state_distribution(policy, d_mu_thresh)
        Q = self.pol_evaluation(policy, Q_thresh)
        er_advantage, distance_sup, distance_mean, target_policy = self.pol_chooser(policy, d_mu_pi, Q)
        convergence_condition = eps / (1 - gamma)
        while er_advantage > convergence_condition and self.iteration < iteration_horizon:
            alfa_num = (1 - gamma) * er_advantage
            alfa_den = 2 * gamma * distance_sup * distance_mean
            alfa_star = alfa_num / alfa_den
            alfa = min(1, alfa_star)
            policy = self.pol_combination(alfa, target_policy, policy)
            d_mu_pi = self.discounted_state_distribution(policy, d_mu_thresh)
            Q = self.pol_evaluation(policy, Q_thresh)
            J_pi = self.pol_performance(policy, Q)

            self._utility_trace(J_pi, alfa_star, er_advantage, convergence_condition,
                                distance_sup, distance_mean, target_policy, prev_target_policy)

            prev_target_policy = target_policy
            er_advantage, distance_sup, distance_mean, target_policy = self.pol_chooser(policy, d_mu_pi, Q)

        return policy


    # implementation of whole algorithm: PolChooser + improvement
    # best of breed implementation with target trick
    def safe_policy_iteration_target_trick(self, initial_policy):

        # initializations
        gamma = self.gamma
        eps = self.eps
        policy = initial_policy
        d_mu_thresh = 0.0001
        Q_thresh = 0.0001
        iteration_horizon = 3000

        # update policy until convergence
        d_mu_pi = self.discounted_state_distribution(policy, d_mu_thresh)
        Q = self.pol_evaluation(policy, Q_thresh)
        er_adv, dist_sup, dist_mean, target = self.pol_chooser(policy, d_mu_pi, Q)
        target_old = target
        convergence_condition = eps / (1 - gamma)
        while er_adv > convergence_condition and self.iteration < iteration_horizon:
            # check if the target has changed: if that is the case compute the bounds
            # for both target and target_old and select the best one
            if not self.pol_equivalence_check(target, target_old):
                er_adv_old, dist_sup_old, dist_mean_old = self.pol_chooser_old(policy, target_old, d_mu_pi, Q)
                bound_old = ((er_adv_old ** 2) * (1 - gamma)) / (4 * gamma * dist_sup_old * dist_mean_old)
                bound = ((er_adv ** 2) * (1 - gamma)) / (4 * gamma * dist_sup * dist_mean)
                # if the target_old is selected update the measures consistently
                if bound_old > bound:
                    er_adv = er_adv_old
                    dist_sup = dist_sup_old
                    dist_mean = dist_mean_old
                    target = target_old
            alfa_star = ((1 - gamma) * er_adv) / (2 * gamma * dist_sup * dist_mean)
            alfa = min(1, alfa_star)
            policy = self.pol_combination(alfa, target, policy)
            d_mu_pi = self.discounted_state_distribution(policy, d_mu_thresh)
            Q = self.pol_evaluation(policy, Q_thresh)
            J_pi = self.pol_performance(policy, Q)

            self._utility_trace(J_pi, alfa_star, er_adv, convergence_condition,
                                dist_sup, dist_mean, target, target_old)

            target_old = target
            er_adv, dist_sup, dist_mean, target = self.pol_chooser(policy, d_mu_pi, Q)

        return policy


    # implementation of whole algorithm: PolChooser + improvement
    # version with target trick without mean value
    def safe_policy_iteration_target_trick_sup(self, initial_policy):

        # initializations
        gamma = self.gamma
        eps = self.eps
        policy = initial_policy
        d_mu_thresh = 0.0001
        Q_thresh = 0.0001
        iteration_horizon = 3000

        # update policy until convergence
        d_mu_pi = self.discounted_state_distribution(policy, d_mu_thresh)
        Q = self.pol_evaluation(policy, Q_thresh)
        er_adv, dist_sup, dist_mean, target = self.pol_chooser(policy, d_mu_pi, Q)
        target_old = target
        convergence_condition = eps / (1 - gamma)
        while er_adv > convergence_condition and self.iteration < iteration_horizon:
            # check if the target has changed: if that is the case compute the bounds
            # for both target and target_old and select the best one
            if not self.pol_equivalence_check(target, target_old):
                er_adv_old, dist_sup_old, dist_mean_old = self.pol_chooser_old(policy, target_old, d_mu_pi, Q)
                bound_old = ((er_adv_old ** 2) * (1 - gamma)) / (4 * gamma * dist_sup_old * dist_sup_old)
                bound = ((er_adv ** 2) * (1 - gamma)) / (4 * gamma * dist_sup * dist_sup)
                # if the target_old is selected update the measures consistently
                if bound_old > bound:
                    er_adv = er_adv_old
                    dist_sup = dist_sup_old
                    dist_mean = dist_mean_old
                    target = target_old
            alfa_star = ((1 - gamma) * er_adv) / (2 * gamma * dist_sup * dist_sup)
            alfa = min(1, alfa_star)
            policy = self.pol_combination(alfa, target, policy)
            d_mu_pi = self.discounted_state_distribution(policy, d_mu_thresh)
            Q = self.pol_evaluation(policy, Q_thresh)
            J_pi = self.pol_performance(policy, Q)

            self._utility_trace(J_pi, alfa_star, er_adv, convergence_condition,
                                dist_sup, dist_mean, target, target_old)

            target_old = target
            er_adv, dist_sup, dist_mean, target = self.pol_chooser(policy, d_mu_pi, Q)

        return policy


    # method which returns the estimated greedy policy
    # and the corresponding expected relative advantage
    def pol_chooser(self, policy, d_mu_pi, Q):

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
    def pol_chooser_old(self, policy, target, d_mu_pi, Q):

        # EXPECTED RELATIVE ADVANTAGE COMPUTATION
        er_advantage = self.er_advantage(target, policy, Q, d_mu_pi)

        # POLICY DISTANCE COMPUTATIONS
        distance_sup = self.pol_infinite_norm(target, policy)
        distance_mean = self.pol_mean_distance(target, policy, d_mu_pi)

        return er_advantage, distance_sup, distance_mean


    # method to compute the performance of a given policy
    def pol_performance(self, policy, Q):

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


    # method to combine linearly target and current with coefficient alfa
    def pol_combination(self, alfa, target, current):

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


    # method to exactly evaluate a policy in terms of q-function
    # it returns Q: dictionary over states indexing array of values
    def pol_evaluation(self, policy, threshold):

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


    # method to compute the greedy policy given the q-function
    def greedy_policy(self, Q):

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
            max = np.amax(q_array)
            # a = np.argwhere(q_array == max).flatten()
            a = np.argwhere(np.abs(q_array - max) < 1e-3).flatten()
            probabilities[a] = float(1) / len(a)
            greedy_policy_rep[s] = probabilities

            # # lexicograph order if more than one greedy
            # a = np.argmax(q_array)
            # probabilities[a] = 1
            # greedy_policy_rep[s] = probabilities

        return greedy_policy_rep


    # method to exactly compute the d_mu_pi
    def discounted_state_distribution(self, policy, threshold):

        # initializations
        mdp = self.mdp
        nS = mdp.nS
        nA = mdp.nA
        h = mdp.horizon
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
            d_mu_next = (1 - gamma) * mu + gamma * dot
            d_mu = d_mu_next
            count = count + 1

        # # value iteration on d_mu: infinite horizon
        # delta = 1
        # while delta > threshold:
        #     delta = 0
        #     dot = np.dot(d_mu, P_pi)
        #     d_mu_next = (1 - gamma) * mu + gamma * dot
        #     max_diff = max(np.absolute(d_mu - d_mu_next))
        #     delta = max(delta, max_diff)
        #     d_mu = d_mu_next

        return d_mu


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


    # method to check the equivalence fo two given policies
    def pol_equivalence_check(self, policy1, policy2):

        policy1_rep = policy1.get_rep()
        policy2_rep = policy2.get_rep()
        outcome = True
        for s in range(self.mdp.nS):
            check = np.array_equiv(policy1_rep[s], policy2_rep[s])
            if not check:
                outcome = False

        return outcome


    # utility method to print the algorithm trace
    # and to collect execution data
    def _utility_trace(self, J_pi, alfa_star, er_advantage, convergence_condition,
                       distance_sup, distance_mean, target, target_old):

        # data collections
        self.iterations.append(self.iteration)
        self.evaluations.append(J_pi)
        self.alfas.append(alfa_star)
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
        check_target = self.pol_equivalence_check(target, target_old)

        # trace print
        print('----------------------')
        print('Alfa star: {0}'.format(alfa_star))
        print('Distance sup: {0}'.format(distance_sup))
        print('Distance mean: {0}'.format(distance_mean))
        print('Condition: {0}'.format(convergence_condition))
        print('Advantage: {0}'.format(er_advantage))
        print('Same target: {0}\n'.format(check_target))
        print('Policy evaluation: {0}'.format(J_pi))
        print('Evaluation average: {0}'.format(average))
        print('Evaluation moving average (10): {0}'.format(moving_average))
        print('Iteration: {0}'.format(self.iteration))

        # iteration update
        self.iteration = self.iteration + 1






    ##########################
    # SAMPLE-BASED ESTIMATIONS
    ##########################

    # method to generate N sample from the discounted state
    # distribution induced by a given policy
    def sampling(self, uniform_policy, policy, N):

        # initializations
        mdp = self.mdp
        nA = mdp.nA
        nS = mdp.nS
        horizon = self.horizon
        gamma = self.gamma

        # STATE SAMPLING PROCEDURE:
        # sample a sequence of states
        # following the current policy
        S = np.zeros(shape=(N,2), dtype=int)
        step_count = 0
        S[0] = [mdp.reset(), step_count]
        # filling the array of samples
        for i in range(1, N):
            # we extract a new sample in the
            # episode until the horizon is met
            if step_count < horizon:
                # we accept a new sample in the episode
                # with probability gamma, reset otherwise
                coin = np.random.binomial(1, gamma)
                if coin == 1:
                    prev_state = S[i-1][0]
                    action = policy.draw_action(prev_state, False)
                    action = np.array([action]).ravel()
                    step = mdp.step(action)
                    state = step[0]
                    done = step[2]
                    # reset if the state is terminal
                    if done:
                        step_count = 0
                        S[i] = [mdp.reset(), step_count]
                    else:
                        step_count = step_count + 1
                        S[i] = [state, step_count]
                else:
                    step_count = 0
                    S[i] = [mdp.reset(), step_count]
            else:
                step_count = 0
                S[i] = [mdp.reset(), step_count]

        return S

    # method to compute the relative advantage for each state
    # and a sample-based estimate of the expected rel. adv.
    def sample_er_advantage(self, target, policy, Q, samples):

        # initializations
        mdp = self.mdp
        nS = mdp.nS

        # relative advantage as a dictionary over states
        A = {s: int() for s in range(nS)}

        # loop to compute the relative advantage for each state
        for s in range(nS):
            pi_t_arr = target[s]
            pi_arr = policy[s]
            q_arr = Q[s]
            A[s] = np.dot(pi_t_arr - pi_arr, q_arr)

        # estimation of the expected relative advantage
        sum = 0
        n_samples = len(samples)
        for sample in samples:
            s = sample[0]
            sum = sum + A[s]
        er_advantage = sum / n_samples

        return er_advantage

    # method to estimate the mean value distance between two given policies
    def sample_pol_mean_distance(self, policy1, policy2, samples):

        # initializations
        policy1_rep = policy1.get_rep()
        policy2_rep = policy2.get_rep()
        mdp = self.mdp
        nA = mdp.nA
        S = samples[:,0]
        n_samples = len(S)

        # mean value computation
        sum_s = np.zeros(n_samples)
        for i in range(n_samples):
            s = S[i]
            sum_a = 0
            for a in range(nA):
                diff_a = policy1_rep[s][a] - policy2_rep[s][a]
                sum_a = sum_a + np.abs(diff_a)
            sum_s[i] = sum_a
        mean = np.mean(sum_s)

        return mean
