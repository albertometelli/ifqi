import numpy as np
import os
import pandas as pd
from ifqi.envs import discrete


class RaceTrackConfigurableEnv(discrete.DiscreteEnv):
    def __init__(self, track_file, initial_configuration, reward_weight=None):

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

        self.horizon = 100
        self.gamma = 0.99

        # nA ---
        self.nA = nA = 5  # 0=KEEP, 1=INCx, 2=INCy, 3=DECx, 4=DECy

        # nS ---
        self.vel = vel = [-2, -1, 0, 1, 2]
        self.nvel = nvel = len(vel)
        self.min_vel, self.max_vel = min(vel), max(vel)
        self.nS = nS = nlin * nvel * nvel  # state=(x,y,vx,vy)

        # isd ---
        isd = np.array(track[tuple(lin.T)] == '1').astype('float64').ravel()
        isd /= isd.sum()

        # P -------------------------
        self.max_psucc = max_psucc = 0.9
        self.min_psucc = min_psucc = 0.1
        self.max_speed = max_speed = 2 * (max(vel) ** 2)

        # P1 and P2 are two extreme models that we aim to combine optimally
        self.P1 = {s: {a: [] for a in range(nA)} for s in range(nS)}
        self.P2 = {s: {a: [] for a in range(nA)} for s in range(nS)}

        # from (vx,vy) to the set of valid actions
        def valid_a(vx, vy):
            actions = [0]  # KEEP
            if vx < 2:
                actions.append(1)  # INCx
            if vx > -2:
                actions.append(2)  # DECx
            if vy < 2:
                actions.append(3)  # INCy
            if vy > -2:
                actions.append(4)  # DECy
            return actions

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

                    valid_actions = valid_a(vx,vy)
                    actions = np.zeros(nA, dtype=int)
                    actions[valid_actions] = valid_actions
                    #Tying to perform an invalid action is like doing nothing

                    for a_index, a_value in enumerate(actions):
                        li1 = self.P1[s][a_index]
                        li2 = self.P2[s][a_index]
                        type = track[x, y]
                        if type == '2':  # if s is goal state
                            li1.append((1.0, s, 0, True))
                            li2.append((1.0, s, 0, True))
                        else:
                            for outcome in [0, 1]:
                                (nx, ny, nvx, nvy) = next_s(x, y, vx, vy, a_value, outcome)
                                ns = self._s_to_i(nx, ny, nvx, nvy)
                                ntype = track[nx, ny]
                                reward = rstate(nx, ny, nvx, nvy, reward_weight)
                                done = (ntype == '2')
                                psucc1 = min_psucc + ((max_psucc - min_psucc) / max_speed) * speed
                                psucc2 = max_psucc - ((max_psucc - min_psucc) / max_speed) * speed
                                prob1 = psucc1 * outcome + (1 - psucc1) * (1 - outcome)
                                prob2 = psucc2 * outcome + (1 - psucc2) * (1 - outcome)
                                li1.append((prob1, ns, reward, done))
                                li2.append((prob2, ns, reward, done))

        # linear combination of P1,P2 with parameter k
        k = initial_configuration
        P = self.model_configuration(k)
        super(RaceTrackConfigurableEnv, self).__init__(nS, nA, P, isd)


    # form state to index
    def _s_to_i(self, x, y, vx, vy):
        s_lin = np.asscalar(np.where((self.lin == (x, y)).all(axis=1))[0])
        index = s_lin * self.nvel * self.nvel + (vx - self.min_vel) * \
                    self.nvel + (vy - self.min_vel)
        return index

    def _load_convert_csv(self, track_file):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + '/tracks/' + track_file + '.csv'
        data_frame = pd.read_csv(path, sep=',', dtype=object)
        data_frame = data_frame.replace(np.nan, ' ', regex=True)
        return data_frame.values

    # linear combination of the extreme models(P1,P2) with parameter k
    def model_configuration(self, k):
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
        return model

    def reset(self):
        rands = discrete.categorical_sample(self.isd, self.np_random)
        [x, y] = self.lin[rands]
        s = self._s_to_i(x, y, 0, 0)
        self.s = np.array([s]).ravel()
        return self.s

    def get_state(self):
        return self.s

    #def step(self, a):
    #    self._step(np.asscalar(a))

