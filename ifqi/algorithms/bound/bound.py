import sys
import numpy as np
import math
import scipy.optimize as opt
from scipy.special import lambertw
import scipy.stats as sts

def variance_pdis_bound(M_2, H, gamma):
    return M_2 * (1 - (gamma ** 2 * M_2) ** H) / (1 - (gamma ** 2 * M_2)) + \
                2 * gamma * M_2 / (1 - gamma) * (1 - (gamma ** 2 * M_2) ** (H - 1)) / (1 - (gamma ** 2 * M_2)) + \
                2 * gamma ** H * M_2 / (1 - gamma) * (1 - (gamma * M_2) ** (H - 1)) / (1 - (gamma * M_2))

def derivative_variance_pdis_bound(M_2, H, gamma):
    return variance_pdis_bound(M_2, H, gamma) / M_2 + M_2 * \
             ((-gamma**(2*H)*H*M_2**(H-1)*(1-gamma**2*M_2) + (1-(gamma**2*M_2)**H)*gamma**2)/(1-gamma**2*M_2)**2 + \
              2*gamma/(1-gamma)*(-gamma**(2*H-2)*(H-1)*M_2**(H-2)*(1-gamma**2*M_2) + (1-(gamma**2*M_2)**(H-1))*gamma**2)/(1-gamma**2*M_2)**2-\
              2*gamma**H/(1-gamma)*(gamma**(H-1)*(H-1)*M_2**(H-2)*(1-gamma*M_2) + (1-(gamma*M_2)**(H-1))*gamma)/(1-gamma*M_2)**2)


class Bound(object):

    def __init__(self, N, delta, gamma, behavioral_policy, target_policy, horizon=np.inf, select_optimal_horizon=False):
        self.N = N
        self.delta = delta
        self.gamma = gamma
        self.horizon = horizon
        self.select_optimal_horizon = select_optimal_horizon
        self.set_policies(behavioral_policy, target_policy)

    def set_policies(self, behavioral_policy, target_policy):
        self.behavioral_policy = behavioral_policy
        self.target_policy = target_policy

    def penalization(self):
        pass

    def gradient_penalization(self):
        pass

    def get_optimal_horizon(self):
        pass


class DummyBound(Bound):

    def set_policies(self, behavioral_policy, target_policy):
        super(DummyBound, self).set_policies(behavioral_policy, target_policy)
        self.H_star = self.horizon

    def penalization(self):
        return 0.

    def gradient_penalization(self):
        return 0.

    def get_optimal_horizon(self):
        return sys.maxsize

class HoeffdingBound(Bound):

    def set_policies(self, behavioral_policy, target_policy):
        super(HoeffdingBound, self).set_policies(behavioral_policy, target_policy)
        self.M_inf = self.target_policy.M_inf(self.behavioral_policy)
        self.M_inf_gradient = self.target_policy.gradient_M_inf(self.behavioral_policy)
        if self.select_optimal_horizon:
            self.H_star = self.get_optimal_horizon()
        else:
            self.H_star = self.horizon

class HoeffdingBoundRatioImportanceWeighting(HoeffdingBound):

    def penalization(self):
        bias = - (self.gamma ** self.H_star - self.gamma ** self.horizon) / (1 - self.gamma)
        var = - (1 - self.gamma ** self.H_star) / \
              (1 - self.gamma) * self.M_inf ** self.H_star \
              * np.sqrt(np.log(1. / self.delta) / self.N)
        return bias + var

    def gradient_penalization(self):
        return - (1 - self.gamma ** self.H_star) / \
              (1 - self.gamma) * self.H_star * self.M_inf ** (self.H_star - 1) \
              * self.M_inf_gradient * np.sqrt(np.log(1. / self.delta) / self.N)

    def get_optimal_horizon(self, approximated=True):
        if approximated:
            f = np.sqrt(np.log(1. / self.delta) / self.N)
            ratio = np.log(1./self.gamma) / (1./self.gamma - 1)
            H_star = 1. / np.log(self.M_inf) * (lambertw(ratio / f * np.exp(ratio)).real - ratio)
            if math.isinf(H_star):
                return sys.maxsize
            return int(round(H_star))
        else:
            raise NotImplementedError()

class HoeffdingBoundPerDecisionRatioImportanceWeighting(HoeffdingBound):
    def penalization(self):
        bias = - (self.gamma ** self.H_star - self.gamma ** self.horizon) / (
        1 - self.gamma)
        var = - (1 - (self.M_inf * self.gamma) ** self.H_star) / \
              (1 - self.M_inf * self.gamma) \
              * np.sqrt(np.log(1. / self.delta) / self.N)
        return bias + var

    def gradient_penalization(self):
        return - (-self.H_star*self.gamma**self.H_star*self.M_inf**(self.H_star-1) + \
                  self.H_star*self.gamma**(self.H_star+1)*self.M_inf**self.H_star - \
                  self.gamma**2*self.M_inf**self.H_star + self.gamma) \
               / (1 - self.gamma * self.M_inf) \
              * self.M_inf_gradient * np.sqrt(np.log(1. / self.delta) / self.N)

    def get_optimal_horizon(self):
        f = np.sqrt(np.log(1. / self.delta) / self.N)
        H_star = 1. / np.log(self.M_inf) * ( \
            np.log((self.gamma * self.M_inf - 1) / np.log(self.gamma * self.M_inf)) + \
            np.log(np.log(self.gamma) / (self.gamma - 1)) - np.log(f))
        if math.isinf(H_star):
            return sys.maxsize
        return int(round(H_star))

class ChebyshevBound(Bound):

    def set_policies(self, behavioral_policy, target_policy):
        super(ChebyshevBound, self).set_policies(behavioral_policy, target_policy)
        self.M_2 = self.target_policy.M_2(self.behavioral_policy)
        self.M_2_gradient = self.target_policy.gradient_M_2(self.behavioral_policy)
        if self.select_optimal_horizon:
            self.H_star = self.get_optimal_horizon()
        else:
            self.H_star = self.horizon

class ChebyshevBoundRatioImportanceWeighting(ChebyshevBound):

    def penalization(self):
        bias = - (self.gamma ** self.H_star - self.gamma ** self.horizon) / (
        1 - self.gamma)
        var = - (1 - self.gamma ** self.H_star) / \
              (1 - self.gamma) * np.sqrt(self.M_2 ** self.H_star / self.N * (1./self.delta - 1))
        return bias + var

    def gradient_penalization(self):
        return - (1 - self.gamma ** self.H_star) / \
              (1 - self.gamma) * self.H_star / 2. * self.M_2 ** (self.H_star / 2. - 1) \
              * self.M_2_gradient * np.sqrt(1. / self.N * (1./self.delta - 1))

    def get_optimal_horizon(self, approximated=True):
        if approximated:
            f = np.sqrt(1. / self.N * (1./self.delta - 1))
            ratio = np.log(1./self.gamma) / (1./self.gamma - 1)
            H_star = 2. / np.log(self.M_2) * (lambertw(ratio / f * np.exp(ratio)).real - ratio)
            H_star = min(H_star, self.horizon)
            return int(H_star)
        else:
            raise NotImplementedError()

class ChebyshevPerDecisionRatioImportanceWeighting(ChebyshevBound):
    def penalization(self):
        bias = - (self.gamma ** self.H_star - self.gamma ** self.horizon) / (
        1 - self.gamma)
        var_bound = variance_pdis_bound(self.M_2, self.H_star, self.gamma)
        var = - (1 - self.gamma ** self.H_star) / \
              (1 - self.gamma) * np.sqrt(var_bound / self.N * (1./self.delta - 1))
        return bias + var

    def gradient_penalization(self):
        var_bound = variance_pdis_bound(self.M_2, self.H_star, self.gamma)
        derivative_var_bound = derivative_variance_pdis_bound(self.M_2, self.H_star, self.gamma)
        return 1./np.sqrt(1./self.delta - 1) * 1./(2*np.sqrt(var_bound)) * derivative_var_bound

    def get_optimal_horizon(self, approximated=True):

        def function(h):
            bias = - (self.gamma ** h - self.gamma ** self.horizon) / (
                       1 - self.gamma)
            var_bound = variance_pdis_bound(self.M_2, h, self.gamma)
            var = - (1 - self.gamma ** h) / \
                  (1 - self.gamma) * np.sqrt(var_bound / self.N * (1. / self.delta - 1))
            return bias + var

        H_star = opt.minimize(function, 1.)
        '''
        Da controllare
        '''
        return H_star


class BernsteinBound(Bound):

    def set_policies(self, behavioral_policy, target_policy):
        super(BernsteinBound, self).set_policies(behavioral_policy, target_policy)
        self.M_2 = self.target_policy.M_2(self.behavioral_policy)
        self.M_2_gradient = self.target_policy.gradient_M_2(self.behavioral_policy)
        self.M_inf = self.target_policy.M_inf(self.behavioral_policy)
        self.M_inf_gradient = self.target_policy.gradient_M_inf(self.behavioral_policy)
        if self.select_optimal_horizon:
            self.H_star = self.get_optimal_horizon()
        else:
            self.H_star = self.horizon

class BernsteinBoundRatioImportanceWeighting(BernsteinBound):

    def penalization(self):
        bias = - (self.gamma ** self.H_star - self.gamma ** self.horizon) / (
        1 - self.gamma)
        var = - (1 - self.gamma ** self.H_star) / \
              (1 - self.gamma) * (2 * self.M_inf ** self.H_star * np.log(1./self.delta) / (3*self.N) +\
                        np.sqrt(2 * self.M_2 ** self.H_star * np.log(1./self.delta) / self.N ))
        return bias + var

    def gradient_penalization(self):
        return - (1 - self.gamma ** self.H_star) / \
              (1 - self.gamma) *(2 * self.H_star * self.M_inf ** (self.H_star-1) * np.log(1./self.delta) / (3*self.N) * self.M_inf_gradient+\
                        np.sqrt(2 * np.log(1./self.delta) / self.N ) * self.M_2 ** (self.H_star/2. - 1) * self.H_star/2. * self.M_2_gradient)

    def get_optimal_horizon(self):
        f = 2 * np.log(1./self.delta) / (3*self.N)
        g = np.sqrt(2 * np.log(1./self.delta) / self.N)
        def function_deriv(h):
            return \
                - self.gamma ** h * np.log(1./self.gamma) \
                + self.gamma ** h * np.log(1./self.gamma) \
                * (self.M_inf ** h * f + self.M_2 ** (h/2) * g) \
                + (1 - self.gamma ** h) * (self.M_inf ** h * np.log(self.M_inf) * f \
                                           + self.M_2 ** (h/2) * .5 * np.log(self.M_2)* g)

        '''
        TODO calcolo di h_sup upper bound on H*
        '''
        h_sup = self.horizon

        H_star = opt.brentq(function_deriv, 1., h_sup)

        if math.isinf(H_star):
            return sys.maxsize
        return int(round(H_star))

class BernsteinBoundPerDecisionRatioImportanceWeighting(BernsteinBound):
    pass


class NormalBound(Bound):

    def set_policies(self, behavioral_policy, target_policy):
        super(NormalBound, self).set_policies(behavioral_policy, target_policy)
        self.M_2 = self.target_policy.M_2(self.behavioral_policy)
        self.M_2_gradient = self.target_policy.gradient_M_2(self.behavioral_policy)
        self.z = sts.norm.ppf(1. - self.delta)

        if self.select_optimal_horizon:
            self.H_star = self.get_optimal_horizon()
        else:
            self.H_star = self.horizon

class NormalBoundRatioImportanceWeighting(NormalBound):

    def penalization(self):
        bias = - (self.gamma ** self.H_star - self.gamma ** self.horizon) / (
        1 - self.gamma)
        var = - (1 - self.gamma ** self.H_star) / \
              (1 - self.gamma) * self.z * np.sqrt(self.M_2 ** self.H_star / self.N)
        return bias + var

    def gradient_penalization(self):
        return - (1 - self.gamma ** self.H_star) / \
              (1 - self.gamma) * self.H_star / 2. * self.M_2 ** (self.H_star / 2. - 1) \
              * self.M_2_gradient * np.sqrt(1. / self.N) * self.z

    def get_optimal_horizon(self, approximated=True):
        if approximated:
            f = np.sqrt(1. / self.N) * self.z
            ratio = np.log(1./self.gamma) / (1./self.gamma - 1)
            H_star = 2. / np.log(self.M_2) * (lambertw(ratio / f * np.exp(ratio)).real - ratio)
            H_star = min(H_star, self.horizon)
            return int(H_star)
        else:
            raise NotImplementedError()