import numpy as np
import os
import pandas as pd
from spmi.envs import discrete


class RaceTrackConfigurableEnv(discrete.DiscreteEnv):
    def __init__(self, track_file, failed_as_keep=False, initial_configuration=None, reward_weight=None):

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
        self.min_vel_nb, self.max_vel_nb = min(vel) + 1, max(vel) - 1
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

        self.max_psuc = max_psuc = 0.9
        self.min_psuc = min_psuc = 0.797
        self.max_psuc2 = max_psuc2 = 0.9
        self.min_psuc2 = min_psuc2 = 0.1
        self.max_pboost = max_pboost = 0.5
        self.min_pboost = min_pboost = 0.
        self.max_speed = max_speed = 2 * (max(vel) ** 2)

        # P_highspeed_noboost, P_lowspeed_noboost, P_highspeed_boost, P_lowspeed_boost
        # are four extreme models that we aim to combine optimally
        self.P_highspeed_noboost = {s: {a: [] for a in range(nA)} for s in range(nS)}
        self.P_lowspeed_noboost = {s: {a: [] for a in range(nA)} for s in range(nS)}
        self.P_highspeed_boost = {s: {a: [] for a in range(nA)} for s in range(nS)}
        self.P_lowspeed_boost = {s: {a: [] for a in range(nA)} for s in range(nS)}

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

        # method to check if the node to be added refers to a new state,
        # if it refers to a new state then append
        # else update the probability of the node with the same state
        def append_if_new(t_list, node):
            found = -1
            for i in range(len(t_list)):
                if t_list[i][1] == node[1]:
                    found = i
                    break
            if found == -1:
                t_list.append(node)
            else:
                t_list[i] = (t_list[i][0] + node[0], t_list[i][1], t_list[i][2], t_list[i][3])

        # populate the transition probability matrix
        # failed actions as random actions
        def fill_p_failed_as_random():

            # filling the value of the vertex models
            for [x, y] in lin:
                for vx in vel:
                    for vy in vel:
                        s = self._s_to_i(x, y, vx, vy)
                        speed = vx ** 2 + vy ** 2

                        valid_actions = self._valid_a(vx, vy)
                        actions = np.zeros(nA, dtype=int)
                        actions[valid_actions] = valid_actions
                        actions_nb = self._valid_a_noboost(vx, vy)
                        # Tying to perform an invalid action is like doing nothing

                        for a_index, a_value in enumerate(actions):
                            li_hs_nb = self.P_highspeed_noboost[s][a_index]
                            li_ls_nb = self.P_lowspeed_noboost[s][a_index]
                            li_hs_b = self.P_highspeed_boost[s][a_index]
                            li_ls_b = self.P_lowspeed_boost[s][a_index]
                            type = track[x, y]
                            if type == '2' or type == '0':  # if s is goal state
                                li_hs_nb.append((1.0, s, 0, True))
                                li_ls_nb.append((1.0, s, 0, True))
                                li_hs_b.append((1.0, s, 0, True))
                                li_ls_b.append((1.0, s, 0, True))
                            else:

                                if a_value in actions_nb:

                                    # SUCCEED ACTION TRANSITION
                                    (nx, ny, nvx, nvy) = next_s(x, y, vx, vy, a_value, 1)
                                    ns = self._s_to_i(nx, ny, nvx, nvy)
                                    ntype = track[nx, ny]
                                    reward = rstate(nx, ny, nvx, nvy, reward_weight)
                                    done = (ntype == '2')
                                    psuc_hs = min_psuc + ((max_psuc - min_psuc) / max_speed) * speed
                                    psuc_ls = max_psuc2 - ((max_psuc2 - min_psuc2) / max_speed) * speed
                                    append_if_new(li_hs_nb, (psuc_hs, ns, reward, done))
                                    append_if_new(li_ls_nb, (psuc_ls, ns, reward, done))
                                    append_if_new(li_hs_b, (psuc_hs, ns, reward, done))
                                    append_if_new(li_ls_b, (psuc_ls, ns, reward, done))

                                    # FAILED ACTION TRANSITIONS
                                    pins_hs = 1 - psuc_hs
                                    pins_ls = 1 - psuc_ls
                                    a_fail = np.copy(actions)
                                    a_fail[a_index] = 0
                                    for i in range(len(a_fail)):
                                        if a_fail[i] not in actions_nb:
                                            a_fail[i] = 0
                                    a_fail = np.unique(a_fail)
                                    if a_value == 0:
                                        a_fail = np.delete(a_fail, 0)
                                    for a in a_fail:
                                        (nx, ny, nvx, nvy) = next_s(x, y, vx, vy, a, 1)
                                        ns = self._s_to_i(nx, ny, nvx, nvy)
                                        ntype = track[nx, ny]
                                        reward = rstate(nx, ny, nvx, nvy, reward_weight)
                                        done = (ntype == '2')
                                        prob_hs = pins_hs / len(a_fail)
                                        prob_ls = pins_ls / len(a_fail)
                                        append_if_new(li_hs_nb, (prob_hs, ns, reward, done))
                                        append_if_new(li_ls_nb, (prob_ls, ns, reward, done))
                                        append_if_new(li_hs_b, (prob_hs, ns, reward, done))
                                        append_if_new(li_ls_b, (prob_ls, ns, reward, done))

                                else:

                                    # SUCCEED ACTION TRANSITIONS
                                    psuc_hs = min_psuc + ((max_psuc - min_psuc) / max_speed) * speed
                                    psuc_ls = max_psuc2 - ((max_psuc2 - min_psuc2) / max_speed) * speed

                                    # no boost state
                                    (nx, ny, nvx, nvy) = next_s(x, y, vx, vy, 0, 1)
                                    ns_nb = self._s_to_i(nx, ny, nvx, nvy)
                                    ntype_nb = track[nx, ny]
                                    done_nb = (ntype_nb == '2')
                                    reward_nb = rstate(nx, ny, nvx, nvy, reward_weight)

                                    # boost state
                                    (nxb, nyb, nvxb, nvyb) = next_s(x, y, vx, vy, a_value, 1)
                                    ns_b = self._s_to_i(nxb, nyb, nvxb, nvyb)
                                    ntype_b = track[nxb, nyb]
                                    done_b = (ntype_b == '2')
                                    reward_b = rstate(nxb, nyb, nvxb, nvyb, reward_weight)

                                    # failure state
                                    (xf, yf) = np.where(track == '0')
                                    xf = np.asscalar(xf)
                                    yf = np.asscalar(yf)
                                    ns_f = self._s_to_i(xf, yf, 0, 0)
                                    done_f = True
                                    reward_f = 0

                                    # high speed boost
                                    prob_b = psuc_hs * max_pboost
                                    prob_f = psuc_hs * max_pboost
                                    prob_nb = psuc_hs * (1 - 2 * max_pboost)
                                    append_if_new(li_hs_b, (prob_b, ns_b, reward_b, done_b))
                                    append_if_new(li_hs_b, (prob_f, ns_f, reward_f, done_f))
                                    append_if_new(li_hs_b, (prob_nb, ns_nb, reward_nb, done_nb))

                                    # low speed boost
                                    prob_b = psuc_ls * max_pboost
                                    prob_f = psuc_ls * max_pboost
                                    prob_nb = psuc_ls * (1 - 2 * max_pboost)
                                    append_if_new(li_ls_b, (prob_b, ns_b, reward_b, done_b))
                                    append_if_new(li_ls_b, (prob_f, ns_f, reward_f, done_f))
                                    append_if_new(li_ls_b, (prob_nb, ns_nb, reward_nb, done_nb))

                                    # high speed no boost
                                    prob_b = psuc_hs * min_pboost
                                    prob_f = psuc_hs * min_pboost
                                    prob_nb = psuc_hs * (1 - 2 * min_pboost)
                                    append_if_new(li_hs_nb, (prob_b, ns_b, reward_b, done_b))
                                    append_if_new(li_hs_nb, (prob_f, ns_f, reward_f, done_f))
                                    append_if_new(li_hs_nb, (prob_nb, ns_nb, reward_nb, done_nb))

                                    # low speed no boost
                                    prob_b = psuc_ls * min_pboost
                                    prob_f = psuc_ls * min_pboost
                                    prob_nb = psuc_ls * (1 - 2 * min_pboost)
                                    append_if_new(li_ls_nb, (prob_b, ns_b, reward_b, done_b))
                                    append_if_new(li_ls_nb, (prob_f, ns_f, reward_f, done_f))
                                    append_if_new(li_ls_nb, (prob_nb, ns_nb, reward_nb, done_nb))


                                    # FAILED ACTION TRANSITIONS
                                    pins_hs = 1 - psuc_hs
                                    pins_ls = 1 - psuc_ls
                                    a_fail = np.copy(actions)
                                    for i in range(len(a_fail)):
                                        if a_fail[i] not in actions_nb:
                                            a_fail[i] = 0
                                    a_fail = np.unique(a_fail)
                                    a_fail = np.delete(a_fail, 0)
                                    for a in a_fail:
                                        (nx, ny, nvx, nvy) = next_s(x, y, vx, vy, a, 1)
                                        ns = self._s_to_i(nx, ny, nvx, nvy)
                                        ntype = track[nx, ny]
                                        reward = rstate(nx, ny, nvx, nvy, reward_weight)
                                        done = (ntype == '2')
                                        prob_hs = pins_hs / len(a_fail)
                                        prob_ls = pins_ls / len(a_fail)
                                        append_if_new(li_hs_nb, (prob_hs, ns, reward, done))
                                        append_if_new(li_ls_nb, (prob_ls, ns, reward, done))
                                        append_if_new(li_hs_b, (prob_hs, ns, reward, done))
                                        append_if_new(li_ls_b, (prob_ls, ns, reward, done))

        # populate the transition probability matrix
        # failed actions as keep actions
        def fill_p_failed_as_keep():

            # filling the value of P_highspeed_noboost and P_lowspeed_noboost
            for [x, y] in lin:
                for vx in vel:
                    for vy in vel:
                        s = self._s_to_i(x, y, vx, vy)
                        speed = vx ** 2 + vy ** 2

                        valid_actions = self._valid_a(vx, vy)
                        actions = np.zeros(nA, dtype=int)
                        actions[valid_actions] = valid_actions
                        # Tying to perform an invalid action is like doing nothing

                        for a_index, a_value in enumerate(actions):
                            li1 = self.P_highspeed_noboost[s][a_index]
                            li2 = self.P_lowspeed_noboost[s][a_index]
                            type = track[x, y]
                            if type == '2':  # if s is goal state
                                li1.append((1.0, s, 0, True))
                                li2.append((1.0, s, 0, True))
                            else:

                                # SUCCEED ACTION TRANSITION
                                (nx, ny, nvx, nvy) = next_s(x, y, vx, vy, a_value, 1)
                                ns = self._s_to_i(nx, ny, nvx, nvy)
                                ntype = track[nx, ny]
                                reward = rstate(nx, ny, nvx, nvy, reward_weight)
                                done = (ntype == '2')
                                psucc1 = min_psuc + ((max_psuc - min_psuc) / max_speed) * speed
                                psucc2 = max_psuc2 - ((max_psuc2 - min_psuc2) / max_speed) * speed
                                prob1 = psucc1
                                prob2 = psucc2
                                li1.append((prob1, ns, reward, done))
                                li2.append((prob2, ns, reward, done))

                                # FAILED ACTION TRANSITION
                                (nx, ny, nvx, nvy) = next_s(x, y, vx, vy, a_value, 0)
                                ns = self._s_to_i(nx, ny, nvx, nvy)
                                ntype = track[nx, ny]
                                reward = rstate(nx, ny, nvx, nvy, reward_weight)
                                done = (ntype == '2')
                                psucc1 = min_psuc + ((max_psuc - min_psuc) / max_speed) * speed
                                psucc2 = max_psuc2 - ((max_psuc2 - min_psuc2) / max_speed) * speed
                                prob1 = 1 - psucc1
                                prob2 = 1 - psucc2
                                li1.append((prob1, ns, reward, done))
                                li2.append((prob2, ns, reward, done))

        # vertex P matrix population
        # default criterion: failed as random
        if failed_as_keep is False:
            fill_p_failed_as_random()
        else:
            fill_p_failed_as_keep()
        # instantiation of model rep for vertex models
        self.P_highspeed_noboost_sas = self.p_sas(self.P_highspeed_noboost)
        self.P_highspeed_noboost_sa = self.p_sa(self.P_highspeed_noboost_sas)
        self.P_lowspeed_noboost_sas = self.p_sas(self.P_lowspeed_noboost)
        self.P_lowspeed_noboost_sa = self.p_sa(self.P_lowspeed_noboost_sas)
        self.P_highspeed_boost_sas = self.p_sas(self.P_highspeed_boost)
        self.P_highspeed_boost_sa = self.p_sa(self.P_highspeed_boost_sas)
        self.P_lowspeed_boost_sas = self.p_sas(self.P_lowspeed_boost)
        self.P_lowspeed_boost_sa = self.p_sa(self.P_lowspeed_boost_sas)
        # linear combination of vertex models with initial parameters
        # default coefficient k_balance: 0.5
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

    # from (vx,vy) to the set of valid actions
    def _valid_a(self, vx, vy):
        actions = [0]  # KEEP
        if vx < self.max_vel:
            actions.append(1)  # INCx
        if vx > self.min_vel:
            actions.append(3)  # DECx
        if vy < self.max_vel:
            actions.append(2)  # INCy
        if vy > self.min_vel:
            actions.append(4)  # DECy
        return actions

    # from (vx,vy) to the set of valid actions (no boost)
    def _valid_a_noboost(self, vx, vy):
        actions = [0]  # KEEP
        if vx < self.max_vel_nb:
            actions.append(1)  # INCx
        if vx > self.min_vel_nb:
            actions.append(3)  # DECx
        if vy < self.max_vel_nb:
            actions.append(2)  # INCy
        if vy > self.min_vel_nb:
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

    def _load_convert_csv(self, track_file):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + '/tracks/' + track_file + '.csv'
        data_frame = pd.read_csv(path, sep=',', dtype=object)
        data_frame = data_frame.replace(np.nan, ' ', regex=True)
        return data_frame.values

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

    # linear combination of the extreme models(P_highspeed_noboost,P_lowspeed_noboost) with parameter k
    def model_configuration(self, k):
        self.k = k
        model = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        for s in range(self.nS):
            for a in range(self.nA):
                li1 = self.P_highspeed_noboost[s][a]
                li2 = self.P_lowspeed_noboost[s][a]
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
        return [self.P_highspeed_noboost_sa, self.P_lowspeed_boost_sa]

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