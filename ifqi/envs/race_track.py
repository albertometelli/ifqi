import numpy as np
from ifqi.envs import discrete


class RaceTrackEnv(discrete.DiscreteEnv):
    def __init__(self, track_file, reward_weight):

        """
        The Race Track environment

        :param track_file: csv file describing the track
        :param reward_weight: input vector to weight the reward basis vector
        """

        # loading of the csv into a matrix
        self.track = track = np.loadtxt(open(track_file, "rb"), delimiter=",", skiprows=1)
        # computation of the track dimensions
        self.nrow, self.ncol = nrow, ncol = track.shape
        # linearized rep of the 2D matrix
        self.lin = lin = np.argwhere(track != ' ' & track != '4')
        # number of valid (x,y) tuple
        self.nlin = nlin = lin.shape[0]

        # nA ---
        nA = 5  # 0=KEEP, 1=INCx, 2=INCy, 3=DECx, 4=DECy

        # nS ---
        nvel = 5
        vel = [-2, -1, 0, 1, 2]
        nS = nlin * nvel * nvel  # state=(x,y,vx,vy)

        # isd ---
        isd = np.array(track[lin] == '1').astype('float64').ravel()
        isd /= isd.sum()

        # P ---
        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        # form state to index
        def s_to_i(x, y, vx, vy):
            s_lin = np.where((lin == (x, y)).all(axis=1))[0]
            index = s_lin * nvel * nvel + vx * nvel + vy
            return index

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
            if x < 0 or x > nrow or y < 0 or y > ncol:
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
            for k in xrange(step, 1-step, step):
                p = k * B + (1-k) * A
                p = np.floor(p)
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

        # filling the value of P
        for [x, y] in lin:
            for vx in vel:
                for vy in vel:
                    s = s_to_i(x, y, vx, vy)
                    speed = vx ** 2 + vy ** 2
                    actions = valid_a(vx,vy)
                    for a in actions:
                        li = P[s][a]
                        type = track[x, y]
                        if type == '2':  # if s is goal state
                            li.append((1.0, s, 0, True))
                        else:
                            for outcome in [0, 1]:
                                (nx, ny, nvx, nvy) = next_s(x, y, vx, vy, a, outcome)
                                ns = s_to_i(nx, ny, nvx, nvy)
                                ntype = track[nx, ny]
                                reward = rstate(nx, ny, nvx, nvy, reward_weight)
                                done = (ntype == '2')
                                if speed < 2:
                                    prob = 0.9 * outcome + 0.1 * (1 - outcome)
                                    li.append((prob, ns, reward, done))
                                else:
                                    prob = 0.6 * outcome + 0.4 * (1 - outcome)
                                    li.append((prob, ns, reward, done))

        super(RaceTrackEnv, self).__init__(nS, nA, P, isd)

    def reset(self):
        rands = discrete.categorical_sample(self.isd, self.np_random)
        [x, y] = self.lin[rands]
        self.s = (x, y, 0, 0)
        return self.s

    def get_state(self):
        return self.s
