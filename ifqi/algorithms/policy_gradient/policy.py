import numpy as np
from gym.utils import seeding
from scipy.stats import multivariate_normal
import numpy.linalg as la

class Policy(object):
    '''
    Abstract class
    '''

    def draw_action(self, state, done):
        pass

    def pdf(self, state, action):
        pass

    def get_dimension(self):
        pass

class ParametricPolicy(Policy):

    def gradient_log(self, state, action, vectorize=True):
        pass

    def get_n_parameters(self):
        pass

class DeterministicPolicyLinearMean(Policy):

    def __init__(self, K):
        self.K = np.array(K, ndmin=2)
        self.dimension = self.K.shape[0]

    def _mean(self, state):
        state = np.array(state, ndmin=1)
        return np.dot(self.K, state)

    def draw_action(self, state, done):
        action = self._mean(state)
        return action

    def pdf(self, state, action):
        if self._mean(state) == action:
            return 1.
        return 0.


class GaussianPolicyLinearMean(ParametricPolicy):
    '''
    '''

    def __init__(self, K, covar):
        self.K = np.array(K, ndmin=2)
        self.covar = np.array(covar, ndmin=2)
        self.inv_covar = la.inv(self.covar)
        self.dimension = self.K.shape[0]
        self.n_parameters = self.K.shape[0] ** 2
        self.seed()

    def _mean(self, state):
        state = np.array(state, ndmin=1)
        return np.dot(self.K, state)

    def draw_action(self, state, done):
        action = self.np_random.multivariate_normal(self._mean(state), self.covar)
        return action

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def set_parameter(self, theta):
        K = self.from_vec_to_param(theta)
        self.K = np.array(K, ndmin=2)

    def from_vec_to_param(self, theta):
        K = theta.reshape(self.K.shape)
        return K

    def from_param_to_vec(self, K):
        K = np.array(K, ndmin=2)
        return K.ravel()

    def get_parameter(self):
        return self.from_param_to_vec(self.K)

    def pdf(self, state, action):
        return multivariate_normal.pdf(action, self._mean(state), self.covar)

    def gradient_log(self, state, action, vectorize=True):
        action = np.array(action, ndmin=1)[:, np.newaxis]
        state = np.array(state, ndmin=1)[:, np.newaxis]
        diff = action - self._mean(state)
        grad_K = la.multi_dot([self.inv_covar, diff, state.T])

        if vectorize:
            return self.from_param_to_vec(grad_K)
        else:
            return grad_K

    def get_dimension(self):
        return self.dimension

    def get_n_parameters(self):
        return self.n_parameters


class GaussianPolicyLinearMeanCholeskyVar(ParametricPolicy):
    '''
    '''

    def __init__(self, K, Lambda):
        self.K = np.array(K, ndmin=2)
        self.Lambda = np.array(Lambda, ndmin=2)
        self.dimension = self.K.shape[0]
        self.n_parameters = self.K.shape[0] ** 2 + (self.Lambda.shape[0] ** 2 + self.Lambda.shape[0]) / 2
        self.covar = np.dot(self.Lambda, self.Lambda.T)
        self.inv_covar = la.inv(self.covar)
        self.seed()

    def _mean(self, state):
        state = np.array(state, ndmin=1)
        return np.dot(self.K, state)

    def draw_action(self, state, done):
        action = self.np_random.multivariate_normal(self._mean(state), self.covar)
        return action

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)


    def set_parameter(self, theta):
        K, Lambda = self.from_vec_to_param(theta)
        self.K = np.array(K, ndmin=2)
        self.Lambda = np.array(Lambda, ndmin=2)
        self.covar = np.dot(self.Lambda, self.Lambda.T)
        self.inv_covar = la.inv(self.covar)
        #self.set_parameter(K, Lambda)

    def from_vec_to_param(self, theta):
        K = theta[:self.K.shape[0] ** 2].reshape(self.K.shape)
        Lambda = np.zeros(self.Lambda.shape)
        Lambda[np.tril_indices(self.Lambda.shape[0])] = theta[self.K.shape[0] ** 2:]
        return K, Lambda

    def from_param_to_vec(self, K, Lambda):
        K = np.array(K, ndmin=2)
        Lambda = np.array(Lambda, ndmin=2)
        return np.concatenate([K.ravel(), Lambda[np.tril_indices(Lambda.shape[0])].ravel()])

    def get_parameter(self):
        return self.from_param_to_vec(self.K,self.Lambda)

    def pdf(self, state, action):
        return multivariate_normal.pdf(action, self._mean(state), self.covar)

    def gradient_log(self, state, action, vectorize=True):
        action = np.array(action, ndmin=1)[:, np.newaxis]
        state = np.array(state, ndmin=1)[:, np.newaxis]
        diff = action - self._mean(state)
        grad_K = la.multi_dot([self.inv_covar, diff, state.T])
        grad_Lambda = -2 * self.dim * np.dot(self.inv_covar, self.Lambda) + 2 * \
                la.multi_dot([self.inv_covar, diff, diff.T, self.inv_covar, self.Lambda])

        if vectorize:
            return self.from_param_to_vec(grad_K, grad_Lambda)
        else:
            return grad_K, grad_Lambda

    def get_dimension(self):
        return self.dimension

    def get_n_parameters(self):
        return self.n_parameters
