import numpy as np
import math

from ifqi.utils.uniform_policy import UniformPolicy
from ifqi.utils.spi_policy import SpiPolicy


class SafePolicyIterator(object):

    def __init__(self, mdp, eps, delta):
        """
        Safe Policy Iterator:
        object that enable the call for spi algorithm

        :param mdp: mdp to solve
        :param eps: accuracy of the spi
        :param delta: correct estimation probability
        """
        self.mdp = mdp
        self.gamma = mdp.gamma
        self.eps = eps
        self.delta = delta
        self.uniform_policy = UniformPolicy(mdp)


    # implementation of whole algorithm: PolChooser + improvement
    def safe_policy_iteration(self, initial_policy):

        # initializations
        gamma = self.gamma
        eps = self.eps
        policy = initial_policy

        # update policy until convergence
        target_policy, er_advantage = self.pol_chooser(policy)
        while er_advantage > (eps / (1 - gamma)):
            distance = self.pol_distance(target_policy, policy)
            alfa_star = (((1 - gamma) ** 3) / gamma) *\
                        (er_advantage / (distance ** 2))
            alfa = min(1, alfa_star)
            policy = self.pol_combination(alfa, target_policy, policy)
            target_policy, er_advantage = self.pol_chooser(policy)

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
        samples = self.sampling(uniform_policy, policy)

        # GREEDY POLICY ESTIMATION
        greedy_policy_rep = {s: [] for s in range(nS)}
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

        # EXPECTED RELATIVE ADVANTAGE ESTIMATION
        # expected relative Q
        sum = 0
        for sample in samples:
            s = int(sample[0])
            a = int(sample[1])
            Qi = sample[2]
            greedy = greedy_policy_rep[s]
            uniform = uniform_policy_rep[s]
            weight = greedy[a] / uniform[a]
            sum = sum + Qi * weight
        er_qfunction = sum / len(samples)
        # expected relative V
        sum = 0
        for sample in samples:
            s = int(sample[0])
            a = int(sample[1])
            Qi = sample[2]
            policy = policy_rep[s]
            uniform = uniform_policy_rep[s]
            weight = policy[a] / uniform[a]
            sum = sum + Qi * weight
        er_vfunction = sum / len(samples)
        # expected relative advantage
        er_advantage = er_qfunction - er_vfunction

        # instantiation of a greedy policy object
        greedy_policy = SpiPolicy(greedy_policy_rep)

        return greedy_policy, er_advantage


    # method to generate N (s,a,Q) samples through a given policy
    def sampling(self, uniform_policy, policy):

        # initializations
        mdp = self.mdp
        nA = mdp.nA
        nS = mdp.nS
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
        N = 1000
        # D = (eps ** 2) * ((1 - gamma) ** 2) * delta
        # N = ((72 * (nA ** 2)) / (eps ** 2)) * (math.log(2 * nPol) + math.log(1 / D))
        # N = int(np.ceil(N))

        # STATE SAMPLING PROCEDURE:
        # sample a sequence of states
        # following the current policy
        S = np.zeros(N, dtype=int)
        S[0] = mdp.reset()
        # filling the array of samples
        for i in range(1, N):
            # we accept a new sample in the episode
            # with probability gamma, reset otherwise
            coin = np.random.binomial(1, gamma)
            if coin == 1:
                prev_state = S[i-1]
                action = policy.draw_action(prev_state, False)
                action = np.array([action]).ravel()
                step = mdp.step(action)
                state = step[0]
                done = step[2]
                # reset if the state is terminal
                if done:
                    S[i] = mdp.reset()
                else:
                    S[i] = state
            else:
                S[i] = mdp.reset()

        # STATE-ACTION SAMPLING PROCEDURE:
        # sample an action, uniformly on the action space,
        # for each state in the array S
        SA = np.zeros(shape=(N,2), dtype=int)
        state = S[0]
        action = uniform_policy.draw_action(state, False)
        action = np.array([action]).ravel()
        SA[0] = [state, action]
        # filling the array of samples
        for i in range(1, N):
            state = S[i]
            action = uniform_policy.draw_action(state, False)
            SA[i] = [state, action]

        # STATE-ACTION-Q SAMPLING PROCEDURE
        # compute the cumulative discounted return
        # for each pair in SA following the current policy
        SAQ = np.zeros(shape=(N,3))
        state = SA[0][0]
        action = SA[0][1]
        action = np.array([action]).ravel()
        Qi = self.evaluation(state, action, policy)
        SAQ[0] = [state, action, Qi]
        # filling the array of samples
        for i in range(1, N):
            state = SA[i][0]
            action = SA[i][1]
            action = np.array([action]).ravel()
            Qi = self.evaluation(state, action, policy)
            SAQ[i] = [state, action, Qi]

        return SAQ


    # method to compute the discounted return
    # for a given (s,a) pair and a given policy
    def evaluation(self, state, action, policy):

        # initializations
        mdp = self.mdp
        gamma = self.gamma
        mdp.set_state(state)
        # first step: take action from state
        step = mdp.step(action)
        state = step[0]
        reward = step[1]
        done = step[2]
        discounted_return = reward
        # loop to reach the end of episode
        # following the given policy
        t = 1
        while not done:
            action = policy.draw_action(state, False)
            action = np.array([action]).ravel()
            step = mdp.step(action)
            state = step[0]
            reward = step[1]
            done = step[2]
            discounted_return = discounted_return + ((gamma ** t) * reward)
            t = t + 1

        return discounted_return


    # method to compute the infinite norm between two given policies
    def pol_distance(self, policy1, policy2):

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