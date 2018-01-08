import numpy as np
import math

from spmi.utils.uniform_policy import UniformPolicy
from spmi.utils.spi_policy import SpiPolicy
from spmi.evaluation.evaluation import collect_episodes


class SPIa(object):

    def __init__(self, mdp, eps, delta):
        """
        Safe Policy Iterator approximate:
        object that enable the call for approximate spi algorithm

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

        ######################
        ######################
        self.count = 0
        self.iteration = 0
        self.iterations = list()
        self.evaluations = list()
        self.advantages = list()
        self.distances_sup = list()
        self.distances_mean = list()
        self.alfas = list()
        ######################

    # implementation of whole algorithm: PolChooser + improvement
    def safe_policy_iteration(self, initial_policy):

        # initializations
        gamma = self.gamma
        eps = self.eps
        policy = initial_policy

        # update policy until convergence
        er_advantage, distance_sup, distance_mean, target_policy = self.pol_chooser(policy)
        convergence_condition = eps / (1 - gamma)
        while er_advantage > convergence_condition and self.iteration < 300:
            alfa_num = (1 - gamma) * er_advantage
            alfa_den = 2 * gamma * distance_sup * distance_mean
            alfa_star = alfa_num / alfa_den
            alfa = min(1, alfa_star)
            policy = self.pol_combination(alfa, target_policy, policy)

            ###############################
            ###############################
            # EVALUATION
            n_episodes = 1000
            episodes = collect_episodes(self.mdp, policy, self.horizon, n_episodes)
            sum_J = 0
            discount = 1
            for count_step in range(len(episodes)):
                step = episodes[count_step]
                sum_J = sum_J + step[2] * discount
                if step[5] != 1:
                    discount = discount * self.gamma
                else:
                    discount = 1
                if step[4] == 1:
                    self.count = self.count + 1
            evaluation = sum_J / n_episodes
            # COLLECTING DATA
            self.iterations.append(self.iteration)
            self.evaluations.append(evaluation)
            self.alfas.append(alfa_star)
            self.advantages.append(er_advantage)
            self.distances_sup.append(distance_sup)
            self.distances_mean.append(distance_mean)
            average = 0
            moving_average = 0
            avg_count = self.iteration
            while avg_count >= 0:
                average = average + self.evaluations[avg_count]
                avg_count = avg_count - 1
            average = average / (self.iteration + 1)
            moving_count = self.iteration
            while moving_count >= 0 and moving_count >= (self.iteration - 9):
                moving_average = moving_average + self.evaluations[moving_count]
                moving_count = moving_count - 1
            moving_average = moving_average / 10
            # SAME TARGET CHECK
            if self.iteration != 0:
                check_target = (target_policy.get_rep() == prev_target_policy.get_rep())
            else:
                check_target = True
            # PRINT
            print('----------------------')
            print('Alfa star: {0}'.format(alfa_star))
            print('Distance sup: {0}'.format(distance_sup))
            print('Distance mean: {0}'.format(distance_mean))
            print('Condition: {0}'.format(convergence_condition))
            print('Advantage: {0}'.format(er_advantage))
            print('Same target: {0}\n'.format(check_target))
            print('Policy evaluation: {0}'.format(evaluation))
            print('Evaluation average: {0}'.format(average))
            print('Evaluation moving average (10): {0}'.format(moving_average))
            print('Episode reaching goal state: {0}\n'.format(self.count / float(10)))
            self.count = 0
            print('Iteration: {0}'.format(self.iteration))
            # PREPARING FOR THE NEXT ITERATION
            self.iteration = self.iteration + 1
            prev_target_policy = target_policy
            ############################

            er_advantage, distance_sup, distance_mean, target_policy = self.pol_chooser(policy)

        return policy


    # method which returns the estimated greedy policy
    # and the corresponding expected relative advantage
    def pol_chooser(self, policy):

        # initializations
        mdp = self.mdp
        nS = mdp.nS
        nA = mdp.nA
        policy_rep = policy.get_rep()
        uniform_policy = self.uniform_policy
        uniform_policy_rep = uniform_policy.get_rep()

        # call the sampling procedure to generate N (s,a,Qi) samples
        samples = self.sampling(uniform_policy, policy, 10000)

        # TARGET POLICY ESTIMATION
        Q = {s: {a: [] for a in range(nA)} for s in range(nS)}
        # loop to order the samples in a [S][A] dictionary
        for sample in samples:
            s = int(sample[0])
            a = int(sample[1])
            Qi = sample[2]
            Q[s][a].append(Qi)
        # loop to average the sample related to the same action
        for s in range(nS):
            for a in range(nA):
                list = Q[s][a]
                n = len(list)
                if n != 0:
                    q = np.sum(list)
                    q = q / n
                    Q[s][a] = q
        # target as a greedy policy (uniform if no information about the (s,a))
        target_policy_rep = self.greedy_policy_estimation(Q)
        # # target as a boltzmann policy (uniform if no information about the (s,a))
        # target_policy_rep = self.boltzmann_policy_estimation(Q)

        # instantiation of a target policy object
        target_policy = SpiPolicy(target_policy_rep)

        # EXPECTED RELATIVE ADVANTAGE ESTIMATION
        # expected relative Q
        sum = 0
        for sample in samples:
            s = int(sample[0])
            a = int(sample[1])
            Qi = sample[2]
            target = target_policy_rep[s]
            uniform = uniform_policy_rep[s]
            weight = target[a] / uniform[a]
            sum = sum + Qi * weight
        er_qfunction = sum / len(samples)
        # expected relative V
        sum = 0
        for sample in samples:
            s = int(sample[0])
            a = int(sample[1])
            Qi = sample[2]
            current_policy = policy_rep[s]
            uniform = uniform_policy_rep[s]
            weight = current_policy[a] / uniform[a]
            sum = sum + Qi * weight
        er_vfunction = sum / len(samples)
        # expected relative advantage
        er_advantage = er_qfunction - er_vfunction

        # POLICY DISTANCE ESTIMATION
        distance_sup = self.pol_infinite_norm(target_policy, policy)
        distance_mean = self.pol_mean_distance(target_policy, policy, samples)

        return er_advantage, distance_sup, distance_mean, target_policy


    # method to estimate the greedy policy given the estimated Q
    def greedy_policy_estimation(self, Q):

        # initializations
        mdp = self.mdp
        nS = mdp.nS
        nA = mdp.nA
        uniform_policy_rep = self.uniform_policy.get_rep()

        greedy_policy_rep = {s: [] for s in range(nS)}
        # loop to give maximum probability to the estimated greedy action,
        # uniform probability if there isn't an estimated greedy action
        for s in range(nS):
            q_array = np.zeros(nA)
            probabilities = np.zeros(nA)
            for a in range(nA):
                if not Q[s][a]:
                    q_array[a] = 0
                else:
                    q_array[a] = np.asscalar(Q[s][a])
            if np.max(q_array) == 0:
                greedy_policy_rep[s] = uniform_policy_rep[s]
            else:
                a = np.argmax(q_array)
                probabilities[a] = 1
                greedy_policy_rep[s] = probabilities

        return greedy_policy_rep


    # method to estimate the boltzmann policy given the estimated Q
    def boltzmann_policy_estimation(self, Q):

        # initializations
        mdp = self.mdp
        nS = mdp.nS
        nA = mdp.nA
        uniform_policy_rep = self.uniform_policy.get_rep()

        tau = 0.01
        boltzmann_policy_rep = {s: [] for s in range(nS)}
        # loop to give maximum probability to the estimated greedy action,
        # uniform probability if there isn't an estimated greedy action
        for s in range(nS):
            q_array = np.zeros(nA)
            probabilities = np.zeros(nA)
            for a in range(nA):
                if not Q[s][a]:
                    q_array[a] = 0
                else:
                    q_array[a] = np.asscalar(Q[s][a])
            if np.max(q_array) == 0:
                boltzmann_policy_rep[s] = uniform_policy_rep[s]
            else:
                q_array = np.exp(q_array / tau)
                den = np.sum(q_array)
                for a in range(nA):
                    probabilities[a] = q_array[a] / den
                boltzmann_policy_rep[s] = probabilities

        return boltzmann_policy_rep


    # method to generate N (s,a,Q) samples through a given policy
    def sampling(self, uniform_policy, policy, N):

        # initializations
        mdp = self.mdp
        nA = mdp.nA
        nS = mdp.nS
        horizon = self.horizon
        gamma = self.gamma
        eps = self.eps
        delta = self.delta
        nPol = nA ** nS


        # COMPUTATION OF N: number of samples needed
        # DEBUG
        # M = 72 * (nA ** 2) / (eps ** 2)
        # L1 = math.log(2 * nPol)
        # L2 = math.log(1 / ((eps ** 2) * ((1 - gamma) ** 2) * delta))
        # N = M * (L1 + L2)


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


        # STATE-ACTION SAMPLING PROCEDURE:
        # sample an action, uniformly on the action space,
        # for each state in the array S
        SA = np.zeros(shape=(N,3), dtype=int)
        state = S[0][0]
        step_count = S[0][1]
        action = uniform_policy.draw_action(state, False)
        action = np.array([action]).ravel()
        SA[0] = [state, action, step_count]
        # filling the array of samples
        for i in range(1, N):
            state = S[i][0]
            action = uniform_policy.draw_action(state, False)
            SA[i] = [state, action, step_count]

        # STATE-ACTION-Q SAMPLING PROCEDURE
        # compute the cumulative discounted return
        # for each pair in SA following the current policy
        SAQ = np.zeros(shape=(N,3))
        state = SA[0][0]
        action = SA[0][1]
        action = np.array([action]).ravel()
        step_count = SA[0][2]
        Qi = self.sa_evaluation(state, action, policy, step_count)
        SAQ[0] = [state, action, Qi]
        # filling the array of samples
        for i in range(1, N):
            state = SA[i][0]
            action = SA[i][1]
            action = np.array([action]).ravel()
            step_count = SA[i][2]
            Qi = self.sa_evaluation(state, action, policy, step_count)
            SAQ[i] = [state, action, Qi]

        return SAQ


    # method to compute the discounted return
    # for a given (s,a) pair and a given policy
    def sa_evaluation(self, state, action, policy, step_count):

        # initializations
        mdp = self.mdp
        gamma = self.gamma
        horizon = self.horizon
        mdp.set_state(state)
        # first step: take action from state
        step = mdp.step(action)
        step_count = step_count + 1
        state = step[0]
        reward = step[1]
        done = step[2] or (step_count >= horizon)
        discounted_return = reward
        # loop to reach the end of episode
        # following the given policy
        t = 1
        while not done:
            action = policy.draw_action(state, False)
            action = np.array([action]).ravel()
            step = mdp.step(action)
            step_count = step_count + 1
            state = step[0]
            reward = step[1]
            done = step[2] or (step_count >= horizon)
            discounted_return = discounted_return + ((gamma ** t) * reward)
            t = t + 1

        return discounted_return


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
    def pol_mean_distance(self, policy1, policy2, samples):

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
