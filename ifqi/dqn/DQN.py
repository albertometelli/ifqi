###################################################
# DQN - Playing Atary with Reinforcement Learning #
###################################################
import numpy as np
from numpy.matlib import repmat
import time
import os

class DQN:

    def __init__(self, env, regr, batch_size = 100, epsilon=1.,epsilon_disc=0.99, discrete_actions = None, action_dim = None,state_dim=None,gamma=0.9):
        self.env = env
        self.sast = np.zeros((0,state_dim*2 + action_dim + 1))
        self.r = np.zeros((0,1))
        self.regr = regr
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.first_time = True
        self.done = True
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.horizon = self.env.horizon
        self.n_step = 0
        self.gamma = gamma
        self.renderize = True
        self.epsilon_disc = epsilon_disc
        self.sameAction = 0
        self.i = 0
        self.random_init = 200

        self.countUndred = 0
        self.countDiscRew = 0
        if isinstance(discrete_actions, np.ndarray):
            if len(discrete_actions.shape) > 1:
                assert discrete_actions.shape[1] == action_dim
                assert discrete_actions.shape[0] > 1, \
                    'Error: at least two actions are required'
                self._actions = discrete_actions
            else:
                assert action_dim == 1
                self._actions = np.array(discrete_actions, dtype='float32').T
        elif isinstance(discrete_actions, list):
            assert len(discrete_actions) > 1, \
                'Error: at least two actions are required'
            self._actions = np.array(
                discrete_actions, dtype='float32').reshape(-1, action_dim)
        else:
            raise ValueError(
                'Supported types for discrete_actions are {np.darray, list')




    def step(self):


        if self.done or self.n_step > self.horizon:
            print "End Of Episode"
            if hasattr(self,"disc_reward"):
                print "discounted reward:" , self.disc_reward
                if self.disc_reward >= 150.:
                    self.countDiscRew += 1
                print "Number of good Ep: ", self.countDiscRew
                print "Number of good landings: ", self.countUndred
            self.n_step = 0
            self.disc_reward = 0
            self.i += 1
            print "episodio:", self.i
            print "epsilon:", self.epsilon
            if hasattr(self,"max_state"):
                print "max_state", self.max_state
                print "min_state", self.min_state

            self.state = np.array(self.env.reset())
            self.epsilon = self.epsilon * self.epsilon_disc

        if self.first_time:
            self.max_state = self.env.env.state #internal state for Acrobot
            self.min_state = self.env.env.state

        if self.sameAction > 0:
            a = self.action
            self.sameAction -= 1
        else:
            if self.first_time or np.random.rand() < self.epsilon:
                if np.random.rand() < 0.1:
                    self.sameAction = np.random.randint(3) * 5 - 1
                a = self.env.action_space.sample()
                self.action = a
                self.first_time = False
            else:
                #matrix
                _, a = self.maxQA(np.array(self.state),np.array([[0.]]))
                a = a[0]

        if self.renderize:
            self.env.render()

        next_state, reward, self.done, info = self.env.step(a)
        if reward >= 100.:
            self.countUndred += 1
            print("Yay! ", reward)

        self.disc_reward += reward
        self.n_step += 1


        #matrix
        new_el = np.array([self.state.tolist() + [a] + \
                 next_state.tolist() + [self.done]])

        self.sast = np.concatenate((self.sast, new_el), axis=0)

        #matrix
        reward = np.array([[reward]])
        self.r = np.concatenate((self.r, reward), axis=0)

        indxs = np.random.permutation(self.sast.shape[0])

        batch_sast = self.sast[indxs, :]
        batch_r = self.r[indxs]

        if self.batch_size < self.sast.shape[0]:
            batch_sast = batch_sast[:self.batch_size,:]
            batch_r = batch_r[:self.batch_size]

        nextstate_idx = self.state_dim + self.action_dim

        self._sa = batch_sast[:, :nextstate_idx]
        self._snext = batch_sast[:, nextstate_idx:-1]
        self._absorbing = batch_sast[:, -1]
        y, _ = self.maxQA(self._snext, self._absorbing)
        #print "y=",y[:10]

        self.regr.fit(self._sa, batch_r + self.gamma * y, nb_epoch=1, verbose=False)

        #internal state for acrobot
        x = np.concatenate((np.matrix(self.max_state), np.abs(np.matrix(self.env.env.state))), axis=0)
        y = np.concatenate((np.matrix(self.min_state), np.abs(np.matrix(self.env.env.state))), axis=0)

        self.max_state = np.max(x, axis=0)
        self.min_state = np.min(y, axis=0)

        self.state = next_state[:]

        return self.done
                      
    def _optimal_action(self,state):
        max_q_a = - np.infty
        max_a = None
        for a in self._actions:
            sample = np.concatenate((state, np.matrix(a)), axis=1)
            q_a = self.regr.predict(sample)
            if q_a >= max_q_a:
                max_a = a
                max_q_a = q_a

        return (max_a, max_q_a)


    def _check_states(self, X):
        """
        Check the correctness of the matrix containing the dataset.
        Args:
            X (numpy.array): the dataset
        Returns:
            The matrix containing the dataset reshaped in the proper way.
        """
        return X.reshape(-1, self.state_dim)

    def save(self,folder,  datasetN):
        directory = os.path.dirname(folder + "/" + "sast" + str(datasetN) + ".npy")
        if not os.path.isdir(directory): os.makedirs(directory)
        np.save(folder + "/" + "sast" + str(datasetN), self.sast)
        np.save(folder + "/" + "r" + str(datasetN), self.r)

    def maxQA(self, states, absorbing, evaluation=False):
        """
        Computes the maximum Q-function and the associated action
        in the provided states.
        Args:
            states (numpy.array): states to be evaluated.
                                  Dimenions: (nsamples x state_dim)
            absorbing (numpy.array): true if the current state is absorbing.
                              Dimensions: (nsamples x 1)
        Returns:
            Q: the maximum Q-value in each state
            A: the action associated to the max Q-value in each state
        """
        new_state = self._check_states(states)
        n_states = new_state.shape[0]
        n_actions = self._actions.shape[0]

        Q = np.zeros((n_states, n_actions))
        abs = np.squeeze(np.asarray(absorbing))
        zero_indx = np.argwhere(abs == 0.)
        one_indx = np.argwhere(abs == 1.)

        for idx in range(n_actions):
            actions = np.matlib.repmat(self._actions[idx], n_states, 1)

            samples = np.concatenate((new_state, actions), axis=1)

            predictions = self.regr.predict(samples)

            Q[zero_indx, idx] = predictions[zero_indx]
            Q[one_indx, idx] = 0
        # compute the maximal action
        amax = np.argmax(Q, axis=1)

        Q_max = np.zeros((n_states, n_actions))
        Q_diff = Q - Q_max
        # store Q-value and action for each state
        rQ, rA = np.zeros(n_states), np.zeros(n_states)
        for idx in range(n_states):
            Q_max[idx,:] = np.array([[Q[idx, amax[idx]]]*n_actions])
            rQ[idx] = Q[idx, amax[idx]]
            rA[idx] = self._actions[amax[idx]]

        assert np.argwhere(Q_max - Q < 0).shape[0] == 0

        return rQ.reshape((rQ.shape[0],1)), rA