import numpy as np

class GradientDescent(object):

    '''
    Abstract class
    '''

    def initialize(self, x0):
        self.x = x0

    def update(self, dx):
        pass

    def get_learning_rate(self, dx):
        pass

    def reduce_learning_rate(self):
        self.learning_rate = self.learning_rate / 2.

    def set_parameter(self, x):
        self.x = x

class VanillaGradient(GradientDescent):

    def __init__(self, learning_rate, ascent=False):
        self.learning_rate = learning_rate
        self.ascent = ascent

    def update(self, dx):
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

    def update(self, dx):
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

    def update(self, dx):
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

    def update(self, dx):
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

    def update(self, dx):
        self.g2 = self.gamma * self.g2 + (1 - self.gamma) * dx ** 2

        if self.ascent:
            self.x += self.learning_rate * dx / np.sqrt(self.g2 + self.eps)
        else:
            self.x -= self.learning_rate * dx / np.sqrt(self.g2 + self.eps)

        return self.x

    def get_learning_rate(self, dx):
        self.g2 = self.gamma * self.g2 + (1 - self.gamma) * dx ** 2
        return self.learning_rate / np.sqrt(self.g2 + self.eps)