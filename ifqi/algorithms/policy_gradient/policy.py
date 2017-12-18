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


class GaussianPolicy(Policy):
    '''
    Gaussian policy with parameter K for the mean and fixed variance
    for any dimension
    TBR
    '''

    def __init__(self, K, covar):
        self.K = np.array(K, ndmin=2)
        self.ndim = self.K.shape[0]
        self.covar = np.array(covar, ndmin=2)
        self.seed()

    def draw_action(self, state, done):
        state = np.array(state, ndmin=1)
        mean = np.dot(self.K, state)
        action = self.np_random.multivariate_normal(mean, self.covar)
        return action

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def set_parameter(self, K, build_gradient=True, build_hessian=True):
        self.K = K

    def pdf(self, state, action):
        mean = np.dot(self.K, state[:, np.newaxis])
        return multivariate_normal.pdf(action, mean.ravel(), self.covar)

    def gradient_log(self, state, action, type_='state-action'):
        if type_ == 'state-action':
            deriv = np.transpose(np.tensordot(np.eye(self.ndim), state, axes=0).squeeze(),
                                 (0, 2, 1))
            mean = np.dot(self.K, state)
            sol = la.solve(self.covar, mean)
            return np.array(np.tensordot(deriv, sol[:, np.newaxis], axes=1).squeeze(), ndmin=2)
        elif type_ == 'list':
            return map(lambda s, a: self.gradient_log(s, a), state, action)

    def hessian_log(self, state, action):
        deriv = np.transpose(
            np.tensordot(np.eye(self.ndim), state, axes=0).squeeze(),
            (0, 2, 1))
        mean = np.dot(self.K, state)
        sol = la.solve(self.covar, mean)
        return np.array(
            np.tensordot(np.tensordot(deriv, deriv, axes=1), sol[:, np.newaxis][:, np.newaxis], axes=1).squeeze(),
            ndmin=2)