import numpy as np
from gym.utils import seeding
from scipy.stats import multivariate_normal
import numpy.linalg as la
from ifqi.utils.tictoc import *

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
    Gaussian Policy with Mean computed as Ks and constant variance
    '''

    def __init__(self, K, covar, max_state=4.0):
        self.K = np.array(K, ndmin=2)
        self.covar = np.array(covar, ndmin=2)
        self.inv_covar = la.inv(self.covar)
        self.dimension = self.K.shape[0]
        self.n_parameters = self.K.shape[0] ** 2
        self.max_state = max_state
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

    def D_inf(self, other, state):
        state = np.array(state, ndmin=1)
        inv_covar_diff = la.inv(other.covar - self.covar)
        mean_diff = np.dot(other.K - self.K, state)
        return la.det(other.covar) ** (self.dimension/2.) / la.det(self.covar) ** (self.dimension/2.) * \
               np.exp(.5 * la.multi_dot([mean_diff, inv_covar_diff, mean_diff]))

    def M_inf(self, other):
        inv_covar_diff = la.inv(other.covar - self.covar)
        param_diff = self.K - other.K
        matrix = la.multi_dot([param_diff.T, inv_covar_diff, param_diff])
        eigvals, _ = la.eigh(matrix)
        max_eigval = eigvals[-1]
        return la.det(other.covar) ** (self.dimension/2.) / la.det(self.covar) ** (self.dimension/2.) * \
               np.exp(.5 * max_eigval * self.max_state ** 2)

    def gradient_M_inf(self, other, vectorize=True):
        inv_covar_diff = la.inv(other.covar - self.covar)
        param_diff = self.K - other.K
        matrix = la.multi_dot([param_diff.T, inv_covar_diff, param_diff])
        eigvals, eigvecs = la.eigh(matrix)
        max_eigval = eigvals[-1]
        max_eigvec = eigvecs[:, -1].ravel()
        deriv = la.multi_dot([np.outer(max_eigvec, max_eigvec), param_diff.T, inv_covar_diff])
        gradient = la.det(other.covar) ** (self.dimension/2.) / la.det(self.covar) **(self.dimension/2.) * \
               np.exp(.5 * max_eigval * self.max_state ** 2) * deriv

        if vectorize:
            return gradient.ravel()
        else:
            return gradient

    def M_2(self, other):
        covar_diff = 2 * other.covar - self.covar
        inv_covar_diff = la.inv(covar_diff)
        param_diff = self.K - other.K
        matrix = la.multi_dot([param_diff.T, inv_covar_diff, param_diff])
        eigvals, _ = la.eigh(matrix)
        max_eigval = eigvals[-1]
        return la.det(other.covar) ** self.dimension / la.det(self.covar) ** (self.dimension / 2.) / \
                la.det(covar_diff) ** (self.dimension / 2.) * \
               np.exp(.5 * max_eigval * self.max_state ** 2)

    def gradient_M_2(self, other, vectorize=True):
        covar_diff = 2 * other.covar - self.covar
        inv_covar_diff = la.inv(covar_diff)
        param_diff = self.K - other.K
        matrix = la.multi_dot([param_diff.T, inv_covar_diff, param_diff])
        eigvals, eigvecs = la.eigh(matrix)
        max_eigval = eigvals[-1]
        max_eigvec = eigvecs[:, -1].ravel()
        deriv = la.multi_dot([np.outer(max_eigvec, max_eigvec), param_diff.T, inv_covar_diff])
        gradient =  la.det(other.covar) ** self.dimension / la.det(self.covar) ** (self.dimension / 2.) / \
                la.det(covar_diff) ** (self.dimension / 2.) * \
                np.exp(.5 * max_eigval * self.max_state ** 2) * deriv

        if vectorize:
            return gradient.ravel()
        else:
            return gradient

class GaussianPolicyLinearMeanCholeskyVar(ParametricPolicy):
    '''
    Gaussian Policy with Mean computed as Ks and variance decomposed
    with Cholesky as Lambda * Lambda.T
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

    def D_inf(self, other, state):
        state = np.array(state, ndmin=1)
        inv_covar_diff = la.inv(other.covar - self.covar)
        mean_diff = np.dot(other.K - self.K, state)
        return la.det(other.covar) ** (self.dimension/2.) / la.det(self.covar) ** (self.dimension/2.) * \
               np.exp(.5 * la.multi_dot([mean_diff, inv_covar_diff, mean_diff]))

    def M_inf(self, other):
        inv_covar_diff = la.inv(other.covar - self.covar)
        param_diff = self.K - other.K
        matrix = la.multi_dot([param_diff.T, inv_covar_diff, param_diff])
        eigvals, _ = la.eigh(matrix)
        max_eigval = eigvals[-1]
        return la.det(other.covar) ** (self.dimension/2.) / la.det(self.covar) ** (self.dimension/2.) * \
               np.exp(.5 * max_eigval * self.max_state ** 2)

    def gradient_M_inf(self, other, vectorize=True):
        inv_covar_diff = la.inv(other.covar - self.covar)
        param_diff = self.K - other.K
        matrix = la.multi_dot([param_diff.T, inv_covar_diff, param_diff])
        eigvals, eigvecs = la.eigh(matrix)
        max_eigval = eigvals[-1]
        max_eigvec = eigvecs[:, -1].ravel()
        deriv = la.multi_dot([np.outer(max_eigvec, max_eigvec), param_diff.T, inv_covar_diff])
        gradient_K = la.det(other.covar) ** (self.dimension/2.) / la.det(self.covar) **(self.dimension/2.) * \
               np.exp(.5 * max_eigval * self.max_state ** 2) * deriv
        gradient_Lambda = -self.dimension * la.det(other.covar) ** (self.dimension/2.) / la.det(self.Lambda) ** self.dimension * \
                          la.inv(self.Lambda).T * np.exp(.5 * max_eigval * self.max_state ** 2) + \
                          - 2 * la.det(other.covar) ** (self.dimension / 2.) / la.det(self.covar) ** (self.dimension / 2.) * \
                          np.exp(.5 * max_eigval * self.max_state ** 2) * inv_covar_diff.T * self.Lambda.T * np.outer(param_diff, param_diff) * self.max_state ** 2 * inv_covar_diff.T

        '''
        Da controllare
        '''

        if vectorize:
            return gradient.ravel()
        else:
            return gradient_K, gradient_Lambda

    def M_2(self, other):
        covar_diff = 2 * other.covar - self.covar
        inv_covar_diff = la.inv(covar_diff)
        param_diff = self.K - other.K
        matrix = la.multi_dot([param_diff.T, inv_covar_diff, param_diff])
        eigvals, _ = la.eigh(matrix)
        max_eigval = eigvals[-1]
        return la.det(other.covar) ** self.dimension / la.det(self.covar) ** (self.dimension / 2.) / \
                la.det(covar_diff) ** (self.dimension / 2.) * \
               np.exp(.5 * max_eigval * self.max_state ** 2)

    def gradient_M_2(self, other, vectorize=True):
        covar_diff = 2 * other.covar - self.covar
        inv_covar_diff = la.inv(covar_diff)
        param_diff = self.K - other.K
        matrix = la.multi_dot([param_diff.T, inv_covar_diff, param_diff])
        eigvals, eigvecs = la.eigh(matrix)
        max_eigval = eigvals[-1]
        max_eigvec = eigvecs[:, -1].ravel()
        deriv = la.multi_dot([np.outer(max_eigvec, max_eigvec), param_diff.T, inv_covar_diff])
        gradient =  la.det(other.covar) ** self.dimension / la.det(self.covar) ** (self.dimension / 2.) / \
                la.det(covar_diff) ** (self.dimension / 2.) * \
                np.exp(.5 * max_eigval * self.max_state ** 2) * deriv

        if vectorize:
            return gradient.ravel()
        else:
            return gradient


class RBFGaussianPolicy(ParametricPolicy):

    def __init__(self,
                 centers,
                 parameters,
                 sigma,
                 radial_basis='gaussian',
                 radial_basis_parameters=None,
                 max_feature_square=0.07**2+1.8**2):

        self.centers = centers
        self.n_centers = self.centers.shape[0]
        self.parameters = parameters.ravel()
        self.dimension = 1
        self.n_parameters = len(self.parameters)
        self.sigma = sigma
        self.max_feature_square = max_feature_square

        if radial_basis == 'gaussian':
            self.radial_basis = lambda x, center: np.exp(-radial_basis_parameters \
                                                         * la.norm(x - center))
        else:
            raise ValueError()

        self.seed()

    def get_dimension(self):
        return self.dimension

    def get_n_parameters(self):
        return self.n_parameters

    def set_parameter(self, parameter):
        self.parameters = parameter.ravel()

    def _mean(self, state):
        rbf = [self.radial_basis(state, self.centers[i])
               for i in range(self.n_centers)]
        mean = np.dot(self.parameters, rbf)
        return mean

    def draw_action(self, state, done):
        action = self._mean(state) + self.np_random.randn() * self.sigma
        return action

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def pdf(self, state, action):
        return np.array(1. / np.sqrt(2 * np.pi * self.sigma ** 2) * np.exp(
            - 1. / 2 * (action - self._mean(state)) ** 2 / self.sigma ** 2),
                        ndmin=1)

    def gradient_log(self, state, action, vectorize=True):
        rbf = [self.radial_basis(state, self.centers[i])
                   for i in range(self.n_centers)]
        mean = np.dot(self.parameters, rbf)
        gradient = (action - mean) / self.sigma ** 2 * np.array(rbf)
        return gradient.ravel()

    def M_inf(self, other):
        param_diff = self.parameters - other.parameters
        return other.sigma / self.sigma * np.exp(param_diff ** 2 * \
            np.exp(self.radial_basis_parameters * 4 * self.max_feature_square) / \
                                 (other.sigma ** 2 - self.sigma ** 2))

    def gradient_M_inf(self, other, vectorize=True):
        param_diff = self.parameters - other.parameters
        return other.sigma / self.sigma * np.exp(param_diff ** 2 * \
            np.exp(self.radial_basis_parameters * 4 * self.max_feature_square) / \
                      (other.sigma ** 2 - self.sigma ** 2)) * 2 * param_diff
    '''
    def M_2(self, other):
        covar_diff = 2 * other.covar - self.covar
        inv_covar_diff = la.inv(covar_diff)
        param_diff = self.K - other.K
        matrix = la.multi_dot([param_diff, inv_covar_diff, param_diff])
        eigvals, _ = la.eigh(matrix)
        max_eigval = eigvals[-1]
        return la.det(other.covar) ** self.dimension / la.det(self.covar) ** (self.dimension / 2.) / \
                la.det(covar_diff) ** (self.dimension / 2.) * \
               np.exp(.5 * max_eigval * self.max_state ** 2)

    def gradient_M_2(self, other, vectorize=True):
        covar_diff = 2 * other.covar - self.covar
        inv_covar_diff = la.inv(covar_diff)
        param_diff = self.K - other.K
        matrix = la.multi_dot([param_diff, inv_covar_diff, param_diff])
        eigvals, eigvecs = la.eigh(matrix)
        max_eigval = eigvals[-1]
        max_eigvec = eigvecs[:, -1].ravel()
        deriv = la.multi_dot([np.outer(max_eigvec, max_eigvec), param_diff.T, inv_covar_diff])
        gradient =  la.det(other.covar) ** self.dimension / la.det(self.covar) ** (self.dimension / 2.) / \
                la.det(covar_diff) ** (self.dimension / 2.) * \
                np.exp(.5 * max_eigval * self.max_state ** 2) * deriv

        if vectorize:
            return gradient.ravel()
        else:
            return gradient
    '''

class GaussianPolicyLinearMeanFeatures(ParametricPolicy):
    '''
        Gaussian Policy with Mean computed as Ks and constant variance
        '''

    def __init__(self, features, parameters, covar, max_feature=1.):
        self.features = features
        self.parameters = np.array(parameters, ndmin=2)
        self.covar = np.array(covar, ndmin=2)
        self.inv_covar = la.inv(self.covar)
        self.dimension = self.parameters.shape[1]
        self.n_parameters = self.parameters.shape[0] * self.parameters.shape[1]
        self.max_feature = max_feature
        self.seed()

    def _mean(self, state):
        feature = self.features(state)
        feature = np.array(feature, ndmin=1)
        return np.dot(self.parameters, feature)

    def draw_action(self, state, done):
        action = self.np_random.multivariate_normal(self._mean(state),
                                                    self.covar)
        return action

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def set_parameter(self, theta):
        self.parameters = theta.reshape(self.parameters.shape)

    def from_vec_to_param(self, theta):
        return theta.reshape(self.parameters.shape)

    def from_param_to_vec(self, parameters):
        return parameters.ravel()

    def get_parameter(self):
        return self.from_param_to_vec(self.parameters)

    def pdf(self, state, action):
        return multivariate_normal.pdf(action, self._mean(state), self.covar)

    def gradient_log(self, state, action, vectorize=True):
        action = np.array(action, ndmin=1)[:, np.newaxis]
        feature = np.array(self.features(state), ndmin=1)[:, np.newaxis]
        diff = action - self._mean(state)
        grad = la.multi_dot([self.inv_covar, diff, feature.T])

        if vectorize:
            return self.from_param_to_vec(grad)
        else:
            return grad

    def get_dimension(self):
        return self.dimension

    def get_n_parameters(self):
        return self.n_parameters

    def M_inf(self, other):
        inv_covar_diff = la.inv(other.covar - self.covar)
        param_diff = self.parameters - other.parameters
        matrix = la.multi_dot([param_diff.T, inv_covar_diff, param_diff])
        eigvals, _ = la.eigh(matrix)
        max_eigval = eigvals[-1]
        return la.det(other.covar) ** (self.dimension / 2.) / la.det(
            self.covar) ** (self.dimension / 2.) * \
               np.exp(.5 * max_eigval * self.max_feature ** 2)

    def gradient_M_inf(self, other, vectorize=True):
        inv_covar_diff = la.inv(other.covar - self.covar)
        param_diff = self.parameters - other.parameters
        matrix = la.multi_dot([param_diff.T, inv_covar_diff, param_diff])
        eigvals, eigvecs = la.eigh(matrix)
        max_eigval = eigvals[-1]
        max_eigvec = eigvecs[:, -1].ravel()
        deriv = la.multi_dot(
            [np.outer(max_eigvec, max_eigvec), param_diff.T, inv_covar_diff])
        gradient = la.det(other.covar) ** (self.dimension / 2.) / la.det(
            self.covar) ** (self.dimension / 2.) * \
                   np.exp(.5 * max_eigval * self.max_feature ** 2) * deriv

        if vectorize:
            return gradient.ravel()
        else:
            return gradient

    def M_2(self, other):
        covar_diff = 2 * other.covar - self.covar
        inv_covar_diff = la.inv(covar_diff)
        param_diff = self.parameters - other.parameters
        matrix = la.multi_dot([param_diff.T, inv_covar_diff, param_diff])
        eigvals, _ = la.eigh(matrix)
        max_eigval = eigvals[-1]
        return la.det(other.covar) ** self.dimension / la.det(self.covar) ** (
        self.dimension / 2.) / \
               la.det(covar_diff) ** (self.dimension / 2.) * \
               np.exp(.5 * max_eigval * self.max_feature ** 2)

    def gradient_M_2(self, other, vectorize=True):
        covar_diff = 2 * other.covar - self.covar
        inv_covar_diff = la.inv(covar_diff)
        param_diff = self.parameters - other.parameters
        matrix = la.multi_dot([param_diff.T, inv_covar_diff, param_diff])
        eigvals, eigvecs = la.eigh(matrix)
        max_eigval = eigvals[-1]
        max_eigvec = eigvecs[:, -1].ravel()
        deriv = la.multi_dot(
            [np.outer(max_eigvec, max_eigvec), param_diff.T, inv_covar_diff])
        gradient = la.det(other.covar) ** self.dimension / la.det(
            self.covar) ** (self.dimension / 2.) / \
                   la.det(covar_diff) ** (self.dimension / 2.) * \
                   np.exp(.5 * max_eigval * self.max_feature ** 2) * deriv

        if vectorize:
            return gradient.ravel()
        else:
            return gradient