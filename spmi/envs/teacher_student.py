import numpy as np
import itertools
from spmi.envs import discrete


class TeacherStudentEnv(discrete.DiscreteEnv):
    def __init__(self, n_literals=2, max_value=1, max_update=1, max_literals_in_examples=3, horizon=10):

        '''
        Constructor
        :param n_literals: number of literals considered in the proble
        :param max_value: literal values are integers ranging in [0, max_value]
        :param max_update: maximum sum of the absolute differences between
                           two consecutive states
        :param max_literals_in_examples: maximum number of literals in an example
        '''

        '''
        The STATE is a list of n_literals components. Eg,
            [5, 4, 1, 0, 1]
        The ACTION is a pair whose first component is the set of the indexes
        of the literals involved in the sum, the second is the value of the sum.
        Eg,
            ({0, 1, 3}, 4)
        meaning that L_0 + L_1 + L_3 = 4
        '''

        self.n_literals = n_literals
        self.max_value = max_value
        self.max_update = max_update
        self.max_literals_in_examples = max_literals_in_examples

        self.nS = self.n_literals ** (self.max_value + 1)

        self.gamma = 0.99
        self.horizon = horizon

        self.isd = np.ones(self.nS) / self.nS
        action_encoded = self._get_all_actions()
        self.nA = len(action_encoded)
        self.encoded_to_index_dict = dict(zip(action_encoded, range(self.nA)))
        self.index_to_encoded_dict = dict(zip(range(self.nA), action_encoded))

        self.P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}


        self._build_P()
        super(TeacherStudentEnv, self).__init__(self.nS, self.nA, self.P, self.isd)


    def _encode_state(self, state):
        index = 0
        for l in state:
            index = index * (self.max_value + 1) + l

        return index

    def _decode_state(self, index):
        state = []
        for i in range(self.n_literals):
            state.append(index % (self.max_value + 1))
            index /= (self.max_value + 1)

        state.reverse()
        return state

    def _encode_action_no_index(self, action):
        index = 0
        for l in action[0]:
            index += 2 ** l
        index = index * self.n_literals * (self.max_value + 1) + action[1]
        return index

    def _encode_action(self, action):
        return self.encoded_to_index_dict[self._encode_action_no_index(action)]

    def _decode_action(self, index):
        index = self.index_to_encoded_dict[index]
        action = (set(), index % (self.n_literals * (self.max_value + 1)))
        index /= (self.n_literals * (self.max_value + 1))
        for l in range(self.n_literals-1, -1, -1):
            if index % 2 == 1:
                action[0].add(l)
            index /= 2

        return action

    def _allowed_state(self, s, s1):
        s = self._decode_state(s)
        s1 = self._decode_state(s1)
        diff = sum(map(lambda x, y: abs(x - y), s, s1))

        return  diff <= self.max_update

    def _consistent(self, a, s):
        a = self._decode_action(a)
        s = self._decode_state(s)

        return sum([s[i] for i in a[0]]) == a[1]

    def _get_all_actions(self):
        action_indexes = []
        for p in range(2, self.max_literals_in_examples + 1):
            literals_combs = map(set, itertools.combinations(range(self.n_literals), p))
            for literals_comb in literals_combs:
                for value in range(0, p * self.max_value + 1):
                    action = (literals_comb, value)
                    action_indexes.append(self._encode_action_no_index(action))
        return action_indexes

    def _build_P(self):

        for s in range(self.nS):
            for a in range(self.nA):
                l = self.P[s][a]
                for s1 in range(self.nS):
                    if self._allowed_state(s, s1):
                        if self._consistent(a, s1):
                            reward = 1
                        else:
                            reward = 0

                        l.append((1., s1, reward, False))

                for i in range(len(l)):
                    l[i] = (l[i][0] / len(l),) + l[i][1:]

    # method to reset the MDP state to an initial one
    def reset(self):
        s = discrete.categorical_sample(self.isd, self.np_random)
        self.s = np.array([s]).ravel()
        return self.s

    # method to get the current MDP state
    def get_state(self):
        return self.s