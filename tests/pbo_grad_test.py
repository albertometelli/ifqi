from __future__ import print_function
from spmi.algorithms.pbo.gradpbo import GradPBO
import numpy as np
from scipy import optimize
import theano
import theano.tensor as T
from sklearn.preprocessing import PolynomialFeatures


def lqr_reg(s, a, theta):
    w1 = theta[0, 0]
    w2 = theta[0, 1]
    v = - w1 ** 2 * s * a - 0.5 * w2 * (a ** 2) - 0.4 * w2 * (s ** 2)
    return v.ravel()


def grad_lqr_reg(s, a, theta):
    w1 = theta[0]
    w2 = theta[1]
    g1 = -2 * w1 * s * a
    g2 = -0.5 * (a ** 2) - 0.4 * (s ** 2)
    return np.array([g1, g2])


def bellmanop(rho, theta):
    rho_s = np.reshape(rho, (theta.shape[1], -1))
    return np.dot(theta, rho_s)


def bellmanop_grad(rho, theta):
    return np.kron(theta.T, np.eye(rho.shape[0], rho.shape[1]))


def empirical_bop(s, a, r, snext, absorbing,
                  all_a, gamma, rho, theta, norm_value=2,
                  incremental=False):
    new_theta = bellmanop(rho, theta)
    if incremental:
        new_theta = theta + new_theta
    qnop = lqr_reg(s, a, new_theta)
    bop = -np.ones(s.shape[0]) * np.inf
    for i in range(s.shape[0]):
        for j in range(all_a.shape[0]):
            qv = lqr_reg(snext[i], all_a[j], theta)
            if qv > bop[i]:
                bop[i] = qv
    v = qnop - r - gamma * bop * (1. - absorbing)
    if norm_value == np.inf:
        err = np.max(np.abs(v))
    elif norm_value % 2 == 0:
        err = np.sum(v ** norm_value) ** (1. / norm_value)
    else:
        err = np.sum(np.abs(v) ** norm_value) ** (1. / norm_value)
    return err, new_theta


def multi_step_ebop(s, a, r, snext, absorbing,
                    all_a, gamma, rho, theta, norm_value=2,
                    incremental=False, steps=1):
    tot_err = 0.0
    t = theta
    for k in range(steps):
        err_k, t = empirical_bop(s, a, r, snext, absorbing,
                                 all_a, gamma, rho, t,
                                 norm_value, incremental)
        tot_err += err_k
    return tot_err, t


class LBPO(object):
    def __init__(self, init_rho):
        self.rho = theano.shared(
            value=np.array(init_rho, dtype=theano.config.floatX),
            borrow=True, name='rho')
        self.theta = T.matrix()
        self.outputs = [T.dot(self.theta, self.rho)]
        self.inputs = [self.theta]
        self.trainable_weights = [self.rho]

    def model(self, theta):
        return T.dot(theta, self.rho)

    def evaluate(self, theta):
        if not hasattr(self, "eval_f"):
            self.eval_f = theano.function(self.inputs, self.outputs[0])
        return self.eval_f(theta)


class LQRRegressor(object):
    def model(self, s, a, omega):
        q = - omega[:, 0] ** 2 * s * a \
            - 0.5 * omega[:, 1] * a * a - 0.4 * omega[:, 1] * s * s
        return q.ravel()


class LQG_PBO(object):
    def __init__(self):
        self.theta = T.matrix()
        # define output for b
        combinations = PolynomialFeatures._combinations(2, 3, False, False)
        n_output_features_ = sum(1 for _ in combinations) + 1
        self.A_b = theano.shared(
            value=np.ones((n_output_features_,), dtype=theano.config.floatX),
            borrow=True, name='A_b')
        self.b_b = theano.shared(value=1.,
                                 borrow=True, name='b_b')

        combinations = PolynomialFeatures._combinations(2, 3, False, False)
        L = [(self.theta[:, 0] ** 0).reshape([-1, 1])]
        for i, c in enumerate(combinations):
            L.append(self.theta[:, c].prod(1).reshape([-1, 1]))
        self.XF3 = T.concatenate(L, axis=1)
        b = (T.dot(self.XF3, self.A_b) + self.b_b).reshape([-1, 1])

        # define output for k
        combinations = PolynomialFeatures._combinations(2, 2, False, False)
        n_output_features_ = sum(1 for _ in combinations) + 1
        self.rho_k = theano.shared(
            value=np.ones((n_output_features_,), dtype=theano.config.floatX),
            borrow=True, name='rho_k')

        combinations = PolynomialFeatures._combinations(2, 2, False, False)
        L = [(self.theta[:, 0] ** 0).reshape([-1, 1])]
        for i, c in enumerate(combinations):
            L.append(self.theta[:, c].prod(1).reshape([-1, 1]))
        self.XF2 = T.concatenate(L, axis=1)
        k = T.dot(self.XF2, self.rho_k).reshape([-1, 1])

        self.outputs = [T.concatenate([b, k], axis=1)]
        self.inputs = [self.theta]
        self.trainable_weights = [self.A_b, self.b_b, self.rho_k]

    def evaluate(self, theta):
        if not hasattr(self, "eval_f"):
            self.eval_f = theano.function(self.inputs, self.outputs[0])
        return self.eval_f(theta)


# F = np.array([[1, 2], [3, 4]])
# print(PolynomialFeatures(3).fit_transform(F))
# lqgpbo = LQG_PBO()
# print(lqgpbo.evaluate(F))

gamma = 0.99
rho = np.array([1., 2., 0., 3.], dtype='float64').reshape(2, 2)
theta = np.array([2., 0.2], dtype='float32').reshape(1, -1)

lbpo = LBPO(rho)  # bellman operator (apx)
q_model = LQRRegressor()  # q-function

s = np.array([1., 2., 3.]).reshape(-1, 1)
a = np.array([0., 3., 4.]).reshape(-1, 1)
nexts = s + 1
r = np.array([-1., -5., 0.])
absorbing = np.array([0., 0., 0.])
discrete_actions = np.array([1, 2, 3]).reshape(-1, 1)
# to be used for maximum estimate

# =================================================================
INCREMENTAL = False
NORM_VAL = 2
ST = 1
gpbo = GradPBO(bellman_model=lbpo, q_model=q_model, steps_ahead=ST,
               discrete_actions=discrete_actions,
               gamma=gamma, optimizer="adam", norm_value=NORM_VAL,
               state_dim=1, action_dim=1, incremental=INCREMENTAL)
gpbo._make_additional_functions()
assert np.allclose(bellmanop(rho, theta), gpbo.F_bellman_operator(theta)), \
    '{}, {}'.format(bellmanop(rho, theta), gpbo.F_bellman_operator(theta))
assert np.allclose(lqr_reg(s, a, theta), gpbo.F_q(s, a, theta))

berr = gpbo.F_bellman_err(s, a, nexts, r, absorbing, theta, discrete_actions)
tv = multi_step_ebop(s, a, r, nexts, absorbing,
                     discrete_actions, gamma, rho, theta,
                     norm_value=NORM_VAL, incremental=INCREMENTAL, steps=ST)[0]
assert np.allclose(berr, tv), '{}, {}'.format(berr, tv)
print(tv)

berr_grad = gpbo.F_grad_bellman_berr(s, a, nexts, r, absorbing,
                                     theta, discrete_actions)
eps = np.sqrt(np.finfo(float).eps)
f = lambda x: multi_step_ebop(s, a, r, nexts, absorbing,
                              discrete_actions, gamma, x, theta,
                              norm_value=NORM_VAL, incremental=INCREMENTAL,
                              steps=ST)[0]
approx_grad = optimize.approx_fprime(rho.ravel(), f, eps).reshape(
    berr_grad[0].shape)
print("{}\n{}".format(berr_grad, approx_grad))
assert np.allclose(berr_grad, approx_grad), '{}, {}'.format(berr_grad,
                                                            approx_grad)
