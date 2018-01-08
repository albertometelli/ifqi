import sys
import numpy as np
import math
import scipy.optimize as opt
from scipy.special import lambertw

class Bound(object):

    def __init__(self, N, delta, gamma, behavioral_policy, target_policy):
        self.N = N
        self.delta = delta
        self.gamma = gamma
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
        self.H_star = self.get_optimal_horizon()

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
        self.H_star = self.get_optimal_horizon()

class HoeffdingBoundRatioImportanceWeighting(HoeffdingBound):

    def penalization(self):
        bias = - self.gamma ** self.H_star / (1 - self.gamma)
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
        bias = - self.gamma ** self.H_star / (1 - self.gamma)
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
        self.H_star = self.get_optimal_horizon()

class ChebyshevBoundRatioImportanceWeighting(ChebyshevBound):

    def penalization(self):
        bias = - self.gamma ** self.H_star / (1 - self.gamma)
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
            if math.isinf(H_star):
                return sys.maxsize
            return int(round(H_star))
        else:
            raise NotImplementedError()

class ChebyshevPerDecisionRatioImportanceWeighting(ChebyshevBound):
    pass

class BernsteinBound(Bound):

    def set_policies(self, behavioral_policy, target_policy):
        super(BernsteinBound, self).set_policies(behavioral_policy, target_policy)
        self.M_2 = self.target_policy.M_2(self.behavioral_policy)
        self.M_2_gradient = self.target_policy.gradient_M_2(self.behavioral_policy)
        self.M_inf = self.target_policy.M_inf(self.behavioral_policy)
        self.M_inf_gradient = self.target_policy.gradient_M_inf(self.behavioral_policy)
        self.H_star = self.get_optimal_horizon()

class BernsteinBoundRatioImportanceWeighting(BernsteinBound):

    def penalization(self):
        bias = - self.gamma ** self.H_star / (1 - self.gamma)
        var = - (1 - self.gamma ** self.H_star) / \
              (1 - self.gamma) * (2 * self.M_inf ** self.H_star * np.log(1./self.delta) / (2*self.N) +\
                        np.sqrt(2 * self.M_2 ** self.H_star * np.log(1./self.delta) / self.N ))
        return bias + var

    def gradient_penalization(self):
        return - (1 - self.gamma ** self.H_star) / \
              (1 - self.gamma) *(2 * self.H_star * self.M_inf ** (self.H_star-1) * np.log(1./self.delta) / (2*self.N) * self.M_inf_gradient+\
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
        h_sup = 20

        H_star = opt.brentq(function_deriv, 1., h_sup)
        print(H_star)
        if math.isinf(H_star):
            return sys.maxsize
        return int(round(H_star))

class BernsteinBoundPerDecisionRatioImportanceWeighting(BernsteinBound):
    pass