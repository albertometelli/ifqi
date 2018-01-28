import numpy as np
import os
import pandas as pd
from spmi.envs import discrete
import copy
import sys
from six import StringIO
from gym import spaces, utils


class RaceTrackConfigurableEnv(discrete.DiscreteEnv):
    def __init__(self, track_file, initial_configuration=None, reward_weight=None):

        """
        The Race Track Configurable environment:
        formulation of Race Track compatible with the configurable MDPs context

        :param track_file: csv file describing the track
        :param initial_configuration: coefficient describing the initial model
        :param reward_weight: input vector to weight the reward basis vector
        """

        # loading of the csv into a matrix
        self.track = track = self._load_convert_csv(track_file)
        # computation of the track dimensions
        self.nrow, self.ncol = nrow, ncol = track.shape
        # linearized rep of the 2D matrix
        self.lin = lin = np.argwhere(np.bitwise_and(track != ' ', track != '4'))
        # number of valid (x,y) tuple
        self.nlin = nlin = lin.shape[0]

        self.horizon = 20
        self.gamma = 0.9

        # nA ---
        self.nA = nA = 5  # 0=KEEP, 1=INCx, 2=INCy, 3=DECx, 4=DECy

        # nS ---
        self.vel = vel = [-2, -1, 0, 1, 2]
        self.nvel = nvel = len(vel)
        self.min_vel, self.max_vel = min(vel), max(vel)
        self.nS = nS = nlin * nvel * nvel  # state=(x,y,vx,vy)

        # isd ---
        mu = np.zeros(nS)
        isd = np.array(track[tuple(lin.T)] == '1').astype('float64').ravel()
        isd /= isd.sum()
        for s in range(nlin):
            if isd[s] != 0:
                [x, y] = lin[s]
                init_s = self._s_to_i(x, y, 0, 0)
                mu[init_s] = 1
        mu /= mu.sum()
        self.mu = mu

        # P -------------------------
        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        self.P_sas = np.zeros(shape=(nS, nA, nS))
        self.P_sa = np.zeros(shape=(nS * nA, nS))

        # min_psucc 0.797 to have combination optimum with track0

        self.max_psucc = max_psucc = 0.9
        self.max_psucc2 = max_psucc2 = 0.9
        self.min_psucc = min_psucc = 0.7
        self.min_psucc2 = min_psucc2 = 0.1
        self.max_speed = max_speed = 2 * (max(vel) ** 2)

        # P1 (high speed) and P2 (low speed) are two extreme models that we aim to combine optimally
        self.P1 = {s: {a: [] for a in range(nA)} for s in range(nS)}
        self.P2 = {s: {a: [] for a in range(nA)} for s in range(nS)}


        # reward computation
        def rstate(x, y, vx, vy, weight):
            if weight is None:
                weight = [1, 0, 0, 0, 0]
            type = track[x, y]
            speed = vx ** 2 + vy ** 2
            isGoal = type == '2'
            isOffroad = type == '3'
            isOnTrack = type == '5'
            isZeroSpeed = speed == 0
            isLowSpeed = speed < 2
            isHighSpeed = speed >= 2
            basis = np.array([isGoal, isOffroad, isZeroSpeed and isOnTrack,
                     isLowSpeed and isOnTrack, isHighSpeed and isOnTrack]).astype('float64')
            reward = np.dot(basis, weight)
            return reward

        # state validity checking
        def check_valid(x, y):
            valid = True
            # check isOutOfBound
            if x < 0 or x >= nrow or y < 0 or y >= ncol:
                valid = False
            # check isBlank
            elif track[x, y] == ' ':
                valid = False
            # check isWall
            elif track[x, y] == '4':
                valid = False
            return valid

        # path validity checking
        def check_path(x1, y1, x2, y2):
            valid = True
            step = 0.1
            A = np.array([x1, y1])
            B = np.array([x2, y2])
            for k in np.arange(step, 1., step):
                p = k * B + (1-k) * A
                p = np.floor(p).astype(int)
                if check_valid(p[0], p[1]):
                    valid = False
            return valid

        # next state computation
        def next_s(x, y, vx, vy, a, outcome):
            if a == 0 or outcome == 0:  # keep or failed action
                nvx = vx
                nvy = vy
                nx = x + nvx
                ny = y + nvy
            else:
                if a == 1:  # increment x
                    nvx = vx + 1
                    nvy = vy
                    nx = x + nvx
                    ny = y + nvy
                elif a == 2:  # increment y
                    nvx = vx
                    nvy = vy + 1
                    nx = x + nvx
                    ny = y + nvy
                elif a == 3:  # decrement x
                    nvx = vx - 1
                    nvy = vy
                    nx = x + nvx
                    ny = y + nvy
                elif a == 4:  # decrement y
                    nvx = vx
                    nvy = vy - 1
                    nx = x + nvx
                    ny = y + nvy
            # check validity of the next state
            if not check_valid(nx, ny):
                return (x, y, 0, 0)
            # check the validity of the path
            elif check_path(x + 0.5, y + 0.5, nx + 0.5, ny):
                return (x, y, 0, 0)
            elif check_path(x + 0.5, y + 0.5, nx, ny + 0.5):
                return (x, y, 0, 0)
            elif check_path(x + 0.5, y + 0.5, nx, ny):
                return (x, y, 0, 0)
            elif check_path(x + 0.5, y + 0.5, nx + 0.5, ny + 0.5):
                return (x, y, 0, 0)
            # return of the validated next state
            return (nx, ny, nvx, nvy)

        # filling the value of P1 and P2
        for [x, y] in lin:
            for vx in vel:
                for vy in vel:
                    s = self._s_to_i(x, y, vx, vy)
                    speed = vx ** 2 + vy ** 2

                    valid_actions = self._valid_a(vx, vy)
                    actions = np.zeros(nA, dtype=int)
                    actions[valid_actions] = valid_actions
                    #Tying to perform an invalid action is like doing nothing

                    for a_index, a_value in enumerate(actions):
                        li1 = self.P1[s][a_index]
                        li2 = self.P2[s][a_index]
                        type = track[x, y]

                        #nss = []

                        if type == '2':  # if s is goal state
                            li1.append((1.0, s, 0, True))
                            li2.append((1.0, s, 0, True))
                            #nss.append(s)
                        else:

                            for outcome in [0, 1]:
                                (nx, ny, nvx, nvy) = next_s(x, y, vx, vy, a_value, outcome)
                                ns = self._s_to_i(nx, ny, nvx, nvy)
                                ntype = track[nx, ny]
                                reward = rstate(nx, ny, nvx, nvy, reward_weight)
                                done = (ntype == '2')
                                psucc1 = min_psucc + ((max_psucc - min_psucc) / max_speed) * speed
                                psucc2 = max_psucc2 - ((max_psucc2 - min_psucc2) / max_speed) * speed
                                prob1 = psucc1 * outcome + (1 - psucc1) * (1 - outcome)
                                prob2 = psucc2 * outcome + (1 - psucc2) * (1 - outcome)
                                if outcome == 1 and ns == li1[0][1]:
                                    li1[0] = (li1[0][0] + prob1, ns, prob1 * reward + (1 - prob1) * li1[0][2], done)
                                    li2[0] = (li2[0][0] + prob2, ns, prob2 * reward + (1 - prob2) * li2[0][2], done)
                                else:
                                    li1.append((prob1, ns, reward, done))
                                    li2.append((prob2, ns, reward, done))
                                #nss.append(ns)
                            '''
                            (nx, ny, nvx, nvy) = next_s(x, y, vx, vy, a_value, 1)
                            ns = self._s_to_i(nx, ny, nvx, nvy)
                            ntype = track[nx, ny]
                            reward = rstate(nx, ny, nvx, nvy, reward_weight)
                            done = (ntype == '2')
                            psucc1 = min_psuc + ((max_psuc - min_psuc) / max_speed) * speed
                            psucc2 = max_psuc2 - ((max_psuc2 - min_psuc2) / max_speed) * speed
                            li1.append((psucc1, ns, reward, done))
                            li2.append((psucc2, ns, reward, done))

                            pins1 = (1 - psucc1) / self.nA
                            pins2 = (1 - psucc2) / self.nA

                            for a in range(self.nA):
                                (nx, ny, nvx, nvy) = next_s(x, y, vx, vy, a, 1)
                                ns = self._s_to_i(nx, ny, nvx, nvy)

                                found = -1
                                for i in range(len(li1)):
                                    if li1[i][1] == ns:
                                        found = i
                                        break

                                if found == -1:
                                    ntype = track[nx, ny]
                                    reward = rstate(nx, ny, nvx, nvy, reward_weight)
                                    done = (ntype == '2')
                                    li1.append((pins1, ns, reward, done))
                                    li2.append((pins2, ns, reward, done))
                                else:
                                    li1[i] = (li1[i][0] + pins1, li1[i][1], li1[i][2], li1[i][3])
                                    li2[i] = (li2[i][0] + pins2, li2[i][1], li2[i][2], li2[i][3])
                                '''
                        '''
                        for [nx, ny] in lin:
                            for nvx in vel:
                                for nvy in vel:
                                    ns = self._s_to_i(nx, ny, nvx, nvy)
                                    if not ns in nss:
                                        reward = rstate(nx, ny, nvx, nvy,
                                                        reward_weight)
                                        ntype = track[nx, ny]
                                        done = (ntype == '2')
                                        li1.append((0., ns, reward, done))
                                        li2.append((0., ns, reward, done))
                        '''

        # instantiation of model rep for P1 and P2
        self.P1_sas = self.p_sas(self.P1)
        self.P1_sa = self.p_sa(self.P1_sas)
        self.P2_sas = self.p_sas(self.P2)
        self.P2_sa = self.p_sa(self.P2_sas)
        # linear combination of P1, P2 with parameter k
        self.initial_configuration = initial_configuration
        if self.initial_configuration is None:
            self.initial_configuration = 0.5
        self.k = self.initial_configuration
        self.model_vector = np.array([self.k, 1 - self.k])
        self.P = P = self.model_configuration(self.k)

        # R ----------
        R = np.zeros(nS)
        # reward vector computation
        for s in range(nS):
            (x, y, vx, vy) = self._i_to_s(s)
            R[s] = rstate(x, y, vx, vy, reward_weight)
        self.R = R

        # call the init method of the super class (Discrete)
        super(RaceTrackConfigurableEnv, self).__init__(nS, nA, P, isd)

    def set_initial_configuration(self, model):
        self.P = model
        self.P_sas = self.p_sas(self.P)
        self.P_sa = self.p_sa(self.P_sas)
        self.k = self.initial_configuration
        self.model_vector = np.array([self.k, 1 - self.k])
        self.P = self.model_configuration(self.k)


    def set_model(self, model):
        self.P = model
        self.P_sas = self.p_sas(self.P)
        self.P_sa = self.p_sa(self.P_sas)

    # from (vx,vy) to the set of valid actions
    def _valid_a(self, vx, vy):
        actions = [0]  # KEEP
        if vx < 2:
            actions.append(1)  # INCx
        if vx > -2:
            actions.append(3)  # DECx
        if vy < 2:
            actions.append(2)  # INCy
        if vy > -2:
            actions.append(4)  # DECy
        return actions

    # form state to index
    def _s_to_i(self, x, y, vx, vy):
        s_lin = np.asscalar(np.where((self.lin == (x, y)).all(axis=1))[0])
        index = s_lin * self.nvel * self.nvel + (vx - self.min_vel) * \
                    self.nvel + (vy - self.min_vel)
        return index

    # form index to state
    def _i_to_s(self, index):
        # vy computation
        vy_off = index % self.nvel
        vy = vy_off + self.min_vel
        # vx computation
        vx_off = (index - vy_off) % (self.nvel * self.nvel)
        vx = (vx_off / self.nvel) + self.min_vel
        # s_lin computation
        s_lin = (index - vx_off - vy_off) / (self.nvel * self.nvel)
        x, y = self.lin[s_lin]
        return (x, y, vx, vy)

    # method to convert the track file csv
    def _load_convert_csv(self, track_file):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + '/tracks/' + track_file + '.csv'
        data_frame = pd.read_csv(path, sep=',', dtype=object)
        data_frame = data_frame.replace(np.nan, ' ', regex=True)
        return data_frame.values

    # linear combination of the extreme models(P1,P2) with parameter k
    def model_configuration(self, k):
        self.k = k
        model = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        for s in range(self.nS):
            for a in range(self.nA):
                li1 = self.P1[s][a]
                li2 = self.P2[s][a]
                for count in range(len(li1)):
                    prob1 = li1[count][0]
                    prob2 = li2[count][0]
                    prob = k * prob1 + (1 - k) * prob2
                    ns = li1[count][1]
                    reward = li1[count][2]
                    done = li1[count][3]
                    model[s][a].append((prob, ns, reward, done))
        # updating the P attributes consistently
        self.P = model
        self.P_sas = self.p_sas(self.P)
        self.P_sa = self.p_sa(self.P_sas)
        return model

    # method to reset the MDP state to an initial one
    def reset(self):
        rands = discrete.categorical_sample(self.isd, self.np_random)
        [x, y] = self.lin[rands]
        s = self._s_to_i(x, y, 0, 0)
        self.s = np.array([s]).ravel()
        return self.s

    # method to get the current MDP state
    def get_state(self):
        return self.s

    # method to get the PHI vector
    def get_phi(self):
        return [self.P1_sa, self.P2_sa]

    # from state index to valid actions
    def get_valid_actions(self, state_index):
        state = self._i_to_s(state_index)
        vx = state[2]
        vy = state[3]
        valid_actions = self._valid_a(vx, vy)
        actions = np.array(valid_actions)
        return actions

    # method to set the MDP current state
    def set_state(self, s):
        if s in self.P:
            self.s = s
        else:
            raise Exception('Invalid state setting')

    # method to set gamma
    def set_gamma(self, gamma):
        self.gamma = gamma

    # method to populate the P_sas
    def p_sas(self, P):

        # initializations
        nS = self.nS
        nA = self.nA

        # instantiation of an SxAxS matrix to collect the probabilities
        P_sas = np.zeros(shape=(nS, nA, nS))

        # loop to fill the probability values
        for s in range(nS):
            for a in range(nA):
                list = P[s][a]
                for s1 in range(nS):
                    prob_sum = 0
                    prob_count = 0
                    for elem in list:
                        if elem[1] == s1:
                            prob_sum = prob_sum + elem[0]
                            prob_count = prob_count + 1
                    if prob_count != 0:
                        p = prob_sum
                        P_sas[s][a][s1] = p

        return P_sas

    # method to populate the P_sa
    def p_sa(self, P_sas):

        # initializations
        nS = self.nS
        nA = self.nA

        P_sa = np.zeros(shape=(nS * nA, nS))
        a = 0
        s = 0
        for sa in range(nS * nA):
            if a == 5:
                a = 0
                s = s + 1
            P_sa[sa] = P_sas[s][a]
            a = a + 1

        return P_sa

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.track.copy().tolist()

        mapper = {'1' : utils.colorize('S', 'blue', highlight=True),
                  '2' : utils.colorize('G', 'green', highlight=True),
                  '3' : ':',
                  '4' : '#',
                  '5' : '.',
                  ' ' : ' '}

        for i in range(len(out)):
            for j in range(len(out[i])):
                out[i][j] = mapper[out[i][j]]

        out = [[c.decode('utf-8') for c in line] for line in out]
        x, y, vx, vy = self._i_to_s(np.asscalar(self.s))

        out[x][y] = utils.colorize(out[x][y], 'yellow', highlight=True)

        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        outfile.write("\n")

        # No need to return anything for human
        if mode != 'human':
            return outfile
