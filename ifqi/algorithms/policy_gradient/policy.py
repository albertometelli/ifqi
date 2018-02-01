import numpy as np
from gym.utils import seeding
from scipy.stats import multivariate_normal
import numpy.linalg as la
from ifqi.utils.tictoc import *
import tensorflow as tf
from ifqi.baselines_adaptor.mlp_policy import MlpPolicy as _MlpPolicy
import baselines.common.tf_util as tf_util
import copy

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
 
    def get_copy(self):
        return copy.deepcopy(self)

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
    '''
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
    '''
    def gradient_M_2(self, other):
        gK = other.covar / (np.sqrt(self.covar * (2*other.covar - self.covar))) * 2 \
            * self.max_state ** 2 / (2*other.covar - self.covar) * (self.K - other.K) * \
            np.exp(la.norm(self.K - other.K) ** 2 * self.max_state ** 2 / (2*other.covar - self.covar))
        #print("gradient M_2 %s" % np.array([np.asscalar(gK), np.asscalar(gL)]))
        return np.array([np.asscalar(gK)])

    def M_2(self, other):
        return np.asscalar(other.covar / (np.sqrt(self.covar * (2*other.covar - self.covar))) *\
               np.exp(la.norm(self.K - other.K) ** 2 * self.max_state ** 2 / (2*other.covar - self.covar)))

class GaussianPolicyLinearMeanCholeskyVar(ParametricPolicy):
    '''
    Gaussian Policy with Mean computed as Ks and variance decomposed
    with Cholesky as Lambda * Lambda.T
    '''

    def __init__(self, K, Lambda, epsilon = 1e-2, max_state=4.0):
        self.K = np.array(K, ndmin=2)
        self.Lambda = np.array(Lambda, ndmin=2)
        self.epsilon = epsilon
        self.dimension = self.K.shape[0]
        self.n_parameters = int(self.K.shape[0] ** 2 + (self.Lambda.shape[0] ** 2 + self.Lambda.shape[0]) / 2)
        self.covar = np.dot(self.Lambda, self.Lambda.T) + np.eye(self.dimension) * epsilon
        self.inv_covar = la.inv(self.covar)
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
        K, Lambda = self.from_vec_to_param(theta)
        self.K = np.array(K, ndmin=2)
        self.Lambda = np.array(Lambda, ndmin=2)
        self.covar = np.dot(self.Lambda, self.Lambda.T) + np.eye(self.dimension) * self.epsilon
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
        grad_Lambda = -2 * self.dimension * np.dot(self.inv_covar, self.Lambda) + 2 * \
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
        raise NotImplementedError()
        inv_covar_diff = la.inv(other.covar - self.covar)
        param_diff = self.K - other.K
        matrix = la.multi_dot([param_diff.T, inv_covar_diff, param_diff])
        eigvals, _ = la.eigh(matrix)
        max_eigval = eigvals[-1]
        return la.det(other.covar) ** (self.dimension/2.) / la.det(self.covar) ** (self.dimension/2.) * \
               np.exp(.5 * max_eigval * self.max_state ** 2)

    def gradient_M_inf(self, other, vectorize=True):
        raise NotImplementedError()
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

    def gradient_M_2(self, other):
        gK = other.covar / (np.sqrt(self.covar * (2*other.covar - self.covar)) ) * 2 \
            * self.max_state ** 2 / (2*other.covar - self.covar) * (self.K - other.K) * \
            np.exp(la.norm(self.K - other.K) ** 2 * self.max_state ** 2 / (2*other.covar - self.covar ))
        gL = (-other.covar  * self.Lambda / (self.covar ** (3./2) * np.sqrt(2*other.covar - self.covar) ) + \
             other.covar * self.Lambda / (np.sqrt(self.covar) * (2*other.covar - self.covar ) ** (3./2)) +\
             2 * self.max_state ** 2 * other.covar * la.norm(self.K - other.K) ** 2 * self.Lambda / (np.sqrt(self.covar) * (2*other.covar - self.covar) ** (5./2))) * \
             np.exp(la.norm(self.K - other.K) ** 2 * self.max_state ** 2 / (2 * other.covar - self.covar))
        #print("gradient M_2 %s" % np.array([np.asscalar(gK), np.asscalar(gL)]))
        return np.array([np.asscalar(gK), np.asscalar(gL)])

    def M_2(self, other):
        M_2 = np.asscalar(other.covar / (np.sqrt(self.covar * (2*other.covar - self.covar))) *\
               np.exp(la.norm(self.K - other.K) ** 2 * self.max_state ** 2 / (2*other.covar - self.covar)))
        return M_2
    '''
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
        gradient_K =  la.det(other.covar) ** self.dimension / la.det(self.covar) ** (self.dimension / 2.) / \
                la.det(covar_diff) ** (self.dimension / 2.) * \
                np.exp(.5 * max_eigval * self.max_state ** 2) * deriv
        gradient_Lambda = -other.covar / (self.covar * np.sqrt(2*other.covar - self.covar)) + other.covar / (2*other.covar - self.covar)**(3./2) + 2 * (.5 * max_eigval * self.max_state ** 2) * other.covar / (2*other.covar - self.covar)**(5./2)
        gradient_Lambda *= np.exp(.5 * max_eigval * self.max_state ** 2)

        #ONLY 1 DIMENSION!!!


        return np.array([np.asscalar(gradient_K), np.asscalar(gradient_Lambda)])
    '''

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

    def pdf_vec(self, states, actions):
        return multivariate_normal.pdf(actions, np.array(map(lambda x: self._mean(x), states)), self.covar)

    def gradient_log(self, state, action, vectorize=True):
        action = np.array(action, ndmin=1)[:, np.newaxis]
        feature = np.array(self.features(state), ndmin=1)[:, np.newaxis]
        diff = action - self._mean(state)
        grad = la.multi_dot([self.inv_covar, diff, feature.T])
        grad_Lambda = -2 * self.dimension * np.dot(self.inv_covar, self.Lambda) + 2 * \
                            la.multi_dot([self.inv_covar, diff, diff.T, self.inv_covar, self.Lambda])

        if vectorize:
            return self.from_param_to_vec(grad, grad_Lambda)
        else:
            return grad, grad_Lambda

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

class GaussianPolicyLinearFeatureMeanCholeskyVar(ParametricPolicy):

    def __init__(self, feature, parameters, Lambda, epsilon=1e-2, max_feature=1.0):
        self.feature = feature
        self.parameters = np.array(parameters, ndmin=1).ravel()
        self.Lambda = np.array(Lambda, ndmin=2)
        self.epsilon = epsilon
        self.dimension = 1
        self.n_parameters = int(self.parameters.shape[0] + (self.Lambda.shape[0] ** 2 + self.Lambda.shape[0]) / 2)
        self.covar = np.dot(self.Lambda, self.Lambda.T) + np.eye(self.dimension) * epsilon
        self.inv_covar = la.inv(self.covar)
        self.max_feature = max_feature
        self.seed()

    def _mean(self, state):
        feature = self.feature(state)
        feature = np.array(feature, ndmin=1)
        return np.array(np.dot(self.parameters, feature), ndmin=1)

    def draw_action(self, state, done):
        action = self.np_random.multivariate_normal(self._mean(state),
                                                    self.covar)
        return action

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def set_parameter(self, theta):
        parameters, Lambda = self.from_vec_to_param(theta)
        self.parameters = parameters
        self.Lambda = np.array(Lambda, ndmin=2)
        self.covar = np.dot(self.Lambda, self.Lambda.T) + np.eye(self.dimension) * self.epsilon
        self.inv_covar = la.inv(self.covar)

    def from_vec_to_param(self, theta):
        parameters = theta[:self.parameters.shape[0]].ravel()
        Lambda = np.zeros(self.Lambda.shape)
        Lambda[np.tril_indices(self.Lambda.shape[0])] = theta[self.parameters.shape[0]:]
        return parameters, Lambda

    def from_param_to_vec(self, parameters, Lambda):
        parameters = np.array(parameters).ravel()
        Lambda = np.array(Lambda, ndmin=2)
        return np.concatenate([parameters, Lambda[np.tril_indices(Lambda.shape[0])].ravel()])

    def get_parameter(self):
        return self.from_param_to_vec(self.parameters, self.Lambda)

    def pdf(self, state, action):
        return multivariate_normal.pdf(action, self._mean(state), self.covar)

    def gradient_log(self, state, action, vectorize=True):
        action = np.array(action, ndmin=1)[:, np.newaxis]
        feature = np.array(self.feature(state), ndmin=1)[:, np.newaxis]
        diff = action - self._mean(state)
        grad = la.multi_dot([self.inv_covar, diff, feature.T])
        grad_Lambda = -2 * self.dimension * np.dot(self.inv_covar, self.Lambda) + 2 * \
                            la.multi_dot([self.inv_covar, diff, diff.T, self.inv_covar, self.Lambda])

        if vectorize:
            return self.from_param_to_vec(grad, grad_Lambda)
        else:
            return grad, grad_Lambda


    def get_dimension(self):
        return self.dimension

    def get_n_parameters(self):
        return self.n_parameters

    def gradient_M_2(self, other):
        gK = other.covar / (np.sqrt(self.covar * (2*other.covar - self.covar)) ) * 2 \
            * self.max_feature ** 2 / (2*other.covar - self.covar) * (self.parameters - other.parameters) * \
            np.exp(la.norm(self.parameters - other.parameters) ** 2 * self.max_feature ** 2 / (2*other.covar - self.covar ))
        gL = (-other.covar  * self.Lambda / (self.covar ** (3./2) * np.sqrt(2*other.covar - self.covar) ) + \
             other.covar * self.Lambda / (np.sqrt(self.covar) * (2*other.covar - self.covar ) ** (3./2)) +\
             2 * self.max_feature ** 2 * other.covar * la.norm(self.parameters - other.parameters) ** 2 * self.Lambda / (np.sqrt(self.covar) * (2*other.covar - self.covar) ** (5./2))) * \
             np.exp(la.norm(self.parameters - other.parameters) ** 2 * self.max_feature ** 2 / (2 * other.covar - self.covar))

        return np.array([gK.ravel(), np.asscalar(gL)])

    def M_2(self, other):
        #print('other covar %s \t this covar %s' % (other.covar, self.covar))
        M_2 = np.asscalar(other.covar / (np.sqrt(self.covar * (2*other.covar - self.covar))) *\
               np.exp(la.norm(self.parameters - other.parameters) ** 2 * self.max_feature ** 2 / (2*other.covar - self.covar)))
        return M_2


class FactGaussianPolicyNNMeanVar(ParametricPolicy):

    #Static attribute
    copy_id = {}

    def __init__(self, name, ob_space, ac_space, hid_size=2,
                 num_hid_layers=2):
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.hid_size = hid_size
        self.num_hid_layers = num_hid_layers
        self.fixed_var = True
        self.name = name
        if not self.name in self.copy_id:
            FactGaussianPolicyNNMeanVar.copy_id[self.name] = 0
        else:
            FactGaussianPolicyNNMeanVar.copy_id[self.name]+=1
        #print('Create ' + self.name + '_'+str(self.copy_id[self.name]))
        self._pol = _MlpPolicy(self.name + '_'+str(self.copy_id[self.name]),
                                 self.ob_space,
                                 self.ac_space,
                                 self.hid_size,
                                 self.num_hid_layers,
                                 gaussian_fixed_var=self.fixed_var)
        self.max_phi = 1 #true for tanh outer activations

    def reshape_s(self,state):
        return np.array(state).reshape(self.ob_space.shape)

    def reshape_a(self,action):
        return np.array(action).reshape(self.ac_space.shape)

    def draw_action(self, state, done):
        return self._pol.act(stochastic=True,ob=self.reshape_s(state))

    def pdf(self, state, action):
        state = self.reshape_s(state)
        action = self.reshape_a(action)
        return self._pol.get_density(state,action)

    def get_dimension(self):
        return len(self.ac_space.shape)

    def gradient_log(self, state, action, vectorize=True):
        state = self.reshape_s(state)
        action = self.reshape_a(action)
        return self._pol.get_score(state,action)

    def get_parameter(self):
        with tf.variable_scope(self.name):
            return self._pol.get_param()

    def get_all_parameters(self):
        return self._pol.get_param()

    def set_parameter(self,param,outer=False):
        self._pol.set_param(param)

    def get_n_parameters(self):
        return  len(self.get_parameter())

    def mean(self,state):
        state = self.reshape_s(state)
        return self._pol.get_mean(state)

    @property
    def covar(self):
        return self._pol.get_std()**2

    @property
    def K(self):
        return self._pol.get_theta()

    def M_2(self,other):
        assert isinstance(other,FactGaussianPolicyNNMeanVar)
        assert len(other.K)==len(self.K)
        M_2 = other.covar / (np.sqrt(self.covar * (2*other.covar - self.covar))) *\
        np.exp(la.norm(self.K - other.K) ** 2 * 2*self.max_phi ** 2 / (2*other.covar - self.covar))
        return np.prod(M_2)

    def gradient_M_2(self,other):
        return 0 #Not implemented!!

    def get_copy(self): 
        a_copy = FactGaussianPolicyNNMeanVar(name = self.name,
                                             ob_space = self.ob_space,
                                             ac_space = self.ac_space,
                                             hid_size = self.hid_size,
                                             num_hid_layers =
                                             self.num_hid_layers)
        a_copy.set_parameter(self.get_all_parameters(),outer=False)
        return a_copy
