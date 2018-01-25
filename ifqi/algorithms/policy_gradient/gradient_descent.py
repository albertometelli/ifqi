import numpy as np
import numpy.linalg as la

class GradientDescent(object):

    '''
    Abstract class
    '''

    def initialize(self, x0):
        self.x = x0

    def update(self, dx, riemann_tensor=None):
        pass

    def get_learning_rate(self, dx):
        pass

    def reduce_learning_rate(self):
        self.learning_rate = self.learning_rate / 2.

    def set_parameter(self, x):
        self.x = x

class ChebychevAdaptiveGradient(GradientDescent):

    def __init__(self, learning_rate, N, delta, gamma, h, ascent=False):
        self.learning_rate = learning_rate
        self.N = N
        self.c = (1 - gamma**h) / (1 - gamma) * np.sqrt(1./delta - 1)
        self.ascent = ascent

    def update(self, dx, riemann_tensor=None):
        learning_rate = self.get_learning_rate(dx, riemann_tensor)

        if self.ascent:
            self.x += learning_rate * self.gradient
        else:
            self.x -= learning_rate * self.gradient

        return self.x

    def get_learning_rate(self, dx, riemann_tensor=None):
        riemann_tensor = np.eye(len(dx)) if riemann_tensor is None else riemann_tensor
        self.gradient = la.solve(riemann_tensor, dx)
        print('Natural gradient: %s' % self.gradient)

        gradient_norm = np.asscalar(np.dot(self.gradient, dx))

        print('Natural gradient norm: %s' % gradient_norm)
        if self.c**2 / self.N <= gradient_norm:
            return self.learning_rate
        return min(self.learning_rate, 1./np.sqrt(self.c**2 / self.N - gradient_norm))

class VanillaGradient(GradientDescent):

    def __init__(self, learning_rate, ascent=False):
        self.learning_rate = learning_rate
        self.ascent = ascent

    def update(self, dx, riemann_tensor=None):
        if self.ascent:
            self.x += self.learning_rate * dx
        else:
            self.x -= self.learning_rate * dx

        return self.x

    def get_learning_rate(self, dx):
        return self.learning_rate


class AnnellingGradient(GradientDescent):

    def __init__(self, learning_rate, ascent=False):
        self.initial_leaning_rate = learning_rate
        self.learning_rate = self.initial_leaning_rate
        self.ascent = ascent
        self.ite = 0

    def update(self, dx, riemann_tensor=None):
        if self.ascent:
            self.x += self.learning_rate * dx
        else:
            self.x -= self.learning_rate * dx
        self.ite += 1
        self.learning_rate = self.initial_leaning_rate / np.sqrt(self.ite)

        return self.x

    def get_learning_rate(self, dx):
        self.ite += 1
        self.learning_rate = self.initial_leaning_rate / np.sqrt(self.ite)
        return self.learning_rate


class Adam(GradientDescent):

    '''
    Kingma, Diederik, and Jimmy Ba.
    "Adam: A method for stochastic optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    '''

    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8, use_correction=False, ascent=False):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.use_correction = use_correction
        self.ascent = ascent

    def initialize(self, x0):
        self.x = x0
        self.m = 0
        self.v = 0
        self.t = 0

    def update(self, dx, riemann_tensor=None):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * dx
        self.v = self.beta2 * self.v + (1 - self.beta2) * (dx ** 2)

        if self.use_correction:
            m = self.m / (1 - self.beta1 ** self.t)
            v = self.v / (1 - self.beta2 ** self.t)
        else:
            m = self.m
            v = self.v

        if self.ascent:
            self.x += self.learning_rate * m / (np.sqrt(v) + self.eps)
        else:
            self.x -= self.learning_rate * m / (np.sqrt(v) + self.eps)

        return self.x


    def get_learning_rate(self, dx):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * dx
        self.v = self.beta2 * self.v + (1 - self.beta2) * (dx ** 2)

        if self.use_correction:
            m = self.m / (1 - self.beta1 ** self.t)
            v = self.v / (1 - self.beta2 ** self.t)
        else:
            m = self.m
            v = self.v

        lr = self.learning_rate * m / (np.sqrt(v) + self.eps) / dx

        return lr

class AdaGrad(GradientDescent):

    def __init__(self, learning_rate, eps=1e-8, ascent=False):
        self.learning_rate = learning_rate
        self.eps = eps
        self.ascent = ascent

    def initialize(self, x0):
        self.x = x0
        self.g2 = 0

    def update(self, dx, riemann_tensor=None):
        self.g2 += dx ** 2

        if self.ascent:
            self.x += self.learning_rate * dx / np.sqrt(self.g2 + self.eps)
        else:
            self.x -= self.learning_rate * dx / np.sqrt(self.g2 + self.eps)

        return self.x

    def get_learning_rate(self, dx):
        self.g2 += dx ** 2
        return self.learning_rate / np.sqrt(self.g2 + self.eps)

class RMSProp(GradientDescent):

    def __init__(self, learning_rate, eps=1e-8, gamma=0.9, ascent=False):
        self.learning_rate = learning_rate
        self.eps = eps
        self.gamma = gamma
        self.ascent = ascent

    def initialize(self, x0):
        self.x = x0
        self.g2 = 0

    def update(self, dx, riemann_tensor=None):
        self.g2 = self.gamma * self.g2 + (1 - self.gamma) * dx ** 2

        if self.ascent:
            self.x += self.learning_rate * dx / np.sqrt(self.g2 + self.eps)
        else:
            self.x -= self.learning_rate * dx / np.sqrt(self.g2 + self.eps)

        return self.x

    def get_learning_rate(self, dx):
        self.g2 = self.gamma * self.g2 + (1 - self.gamma) * dx ** 2
        return self.learning_rate / np.sqrt(self.g2 + self.eps)