import numpy.linalg as la
import numpy as np
from ifqi.evaluation import evaluation
import copy
from reward_space.policy_gradient.gradient_descent import *

class PolicyGradientLearner(object):

    '''
    This class performs policy search exploiting the policy gradient. Policy
    parameters are optimized using simple gradient ascent.
    '''

    def __init__(self,
                 mdp,
                 policy,
                 lrate=0.01,
                 estimator='reinforce',
                 baseline_type='vectorial',
                 gradient_updater='vanilla',
                 max_iter_eval=100,
                 tol_eval=-1.,
                 max_iter_opt=100,
                 tol_opt=-1.,
                 verbose=0,
                 state_index=0,
                 action_index=1,
                 reward_index=2):

        '''
        Constructor

        param mdp: the Markov Decision Process
        param policy: the policy whose parameters are optimized
        param lrate: the learning rate
        param lrate_decay: the decay factor, currently not implemented
        param estimator: the name of the gradient estimtor to use, currently
                         only 'reinforce' is supported
        param max_iter_eval: the maximum number of iteration for gradient
                             estimation
        param tol_eval: the estimation stops when norm-2 of the gradient
                        increment is below tol_eval
        param max_iter_opt: the maximum number of iteration for gradient
                             optimization
        param tol_opt: the optimization stops when norm-2 of the gradient
                       increment is below tol_eval
        param verbose: verbosity level
        '''

        self.mdp = mdp
        self.policy = copy.deepcopy(policy)
        self.lrate = lrate
        self.max_iter_eval = max_iter_eval
        self.tol_eval = tol_eval
        self.max_iter_opt = max_iter_opt
        self.tol_opt = tol_opt
        self.verbose = verbose
        self.state_index= state_index
        self.action_index = action_index
        self.reward_index = reward_index
        self.baseline_type = baseline_type

        if estimator == 'reinforce':
            self.estimator = ReinforceGradientEstimator(self.mdp,
                                                        self.policy,
                                                        self.tol_eval,
                                                        self.max_iter_eval,
                                                        self.verbose == 2,
                                                        self.state_index,
                                                        self.action_index,
                                                        self.reward_index)
        elif estimator == 'gpomdp':
            self.estimator = GPOMDPGradientEstimator(self.mdp,
                                                        self.policy,
                                                        self.tol_eval,
                                                        self.max_iter_eval,
                                                        self.verbose == 2,
                                                        self.state_index,
                                                        self.action_index,
                                                        self.reward_index)
        else:
            raise NotImplementedError()

        if gradient_updater == 'vanilla':
            self.gradient_updater = VanillaGradient(lrate, ascent=True)
        elif gradient_updater == 'adam':
            self.gradient_updater = Adam(lrate, ascent=True)

    def optimize(self, theta0, return_history=False):
        '''
        This method performs simple gradient ascent optimization of the policy
        parameters.

        param theta0: the initial value of the parameter
        '''

        ite = 0
        theta = np.copy(theta0)
        self.gradient_updater.initialize(theta)

        self.policy.set_parameter(theta)
        self.estimator.set_policy(self.policy)

        if self.verbose >= 1:
            print('Policy gradient: starting optimization...')

        gradient, avg_return = self.estimator.estimate(baseline_type=self.baseline_type)

        if return_history:
            history = [[np.copy(theta), avg_return, gradient]]

        gradient_norm = la.norm(gradient)

        if self.verbose >= 1:
            print('Ite %s: return %s - gradient norm %s' % (ite, avg_return, gradient_norm))

        while ite < self.max_iter_opt and gradient_norm > self.tol_opt:

            print(theta)
            print(gradient)
            theta = self.gradient_updater.update(gradient) #Gradient ascent update

            self.policy.set_parameter(theta)
            self.estimator.set_policy(self.policy)
            gradient, avg_return  = self.estimator.estimate(baseline_type=self.baseline_type)
            if return_history:
                history.append([np.copy(theta), avg_return, gradient])

            gradient_norm = la.norm(gradient)
            ite += 1

            if self.verbose >= 1:
                print('Ite %s: return %s - gradient norm %s' % (
                ite, avg_return, gradient_norm))

        if return_history:
            return theta, history
        else:
            return theta

class GradientEstimator(object):
    '''
    Abstract class for gradient estimators
    '''

    eps = 1e-24  # Tolerance used to avoid divisions by zero

    def __init__(self,
                 mdp,
                 policy,
                 tol=1e-5,
                 max_iter=100,
                 verbose=True,
                 state_index=0,
                 action_index=1,
                 reward_index=2):
        '''
        Constructor

        param mdp: the Markov Decision Process
        param policy: the policy used to collect samples
        param tol: the estimation stops when norm-2 of the gradient increment is
                   below tol
        param max_iter: the maximum number of iterations for the algorithm
        param verbose: whether to display progressing messages
        '''

        self.mdp = mdp
        self.policy = policy
        self.tol = tol
        self.max_iter = max_iter
        self.dim = policy.get_dim()
        self.verbose = verbose
        self.state_index= state_index
        self.action_index = action_index
        self.reward_index = reward_index
        self.gamma = mdp.gamma


    def set_policy(self, policy):
        self.policy = policy

    def estimate(self, reward=None):
       pass


class ReinforceGradientEstimator(GradientEstimator):

    '''
    This class implements the Reinforce algorithm for gradient estimation with
    element wise optimal baseline

    Williams, Ronald J. "Simple statistical gradient-following algorithms for
    connectionist reinforcement learning." Machine learning 8.3-4 (1992): 229-256.
    '''

    def estimate(self, baseline_type='vectorial'):
        '''
        This method performs gradient estimation with Reinforce algorithm.

        param use_baseline: whether to use the optimal baseline to estimate
                            the gradient
        return: the estimated gradient
        '''

        if self.verbose:
            print('\tReinforce: starting estimation...')

        ite = 0
        gradient_increment = np.inf
        gradient_estimate = np.inf

        #vector of the trajectory returns
        traj_returns = np.ndarray((0, 1))

        # matrix of the tragectory log policy gradient
        traj_log_gradients = np.ndarray((0, self.dim))

        while ite < self.max_iter and gradient_increment > self.tol:
            ite += 1

            # Collect a trajectory
            traj = evaluation.collect_episode(self.mdp, self.policy)

            # Compute the trajectory return
            traj_return = np.dot(traj[:, self.reward_index], self.gamma ** np.arange(traj.shape[0]))
            traj_returns = np.vstack([traj_returns, [traj_return]])

            # Compute the trajectory log policy gradient
            traj_log_gradient = 0.
            for i in range(traj.shape[0]):
                traj_log_gradient += self.policy.gradient_log(traj[i, self.state_index], traj[i, self.action_index])
            traj_log_gradients = np.vstack([traj_log_gradients, [traj_log_gradient]])


            # Compute the optimal baseline
            if baseline_type == 'vectorial':
                baseline = np.sum(traj_log_gradients ** 2 * traj_returns, axis=0) \
                        / (self.eps + np.sum(traj_log_gradients ** 2, axis=0))

            elif baseline_type == 'scalar':
                baseline = np.sum(la.norm(traj_log_gradients ** 2 * traj_returns, axis=1), axis=0) \
                           / (self.eps + np.sum(la.norm(traj_log_gradients ** 2, axis=1), axis=0))
            else:
                baseline = 0.

            # Compute the gradient estimate
            old_gradient_estimate = gradient_estimate
            gradient_estimate = 1. / ite * np.sum(traj_log_gradients * (traj_returns - baseline), axis=0)

            # Compute the gradient increment
            gradient_increment = la.norm(
                gradient_estimate - old_gradient_estimate)

            if self.verbose:
                print('\tIteration %s return %s gradient_norm %s gradient_increment %s' % (ite, traj_return, la.norm(gradient_estimate), gradient_increment))

        return gradient_estimate, np.mean(traj_returns)

class GPOMDPGradientEstimator(GradientEstimator):

    def estimate(self, baseline_type='vectorial'):

        if self.verbose:
            print('\tReinforce: starting estimation...')

        ite = 0
        gradient_increment = np.inf
        gradient_estimate = np.inf

        #vector of the trajectory returns
        traj_returns = np.ndarray((0, self.mdp.horizon))

        # matrix of the tragectory log policy gradient
        traj_log_gradients = np.ndarray((0, self.mdp.horizon, self.dim))

        #vectors of the lenghts of trajectories
        horizons = []

        while ite < self.max_iter and gradient_increment > self.tol:
            ite += 1

            # Collect a trajectory
            traj = evaluation.collect_episode(self.mdp, self.policy)

            #Compute lenght
            horizons.append(traj.shape[0])

            # Compute the trajectory return
            traj_return = np.zeros(self.mdp.horizon)
            traj_return[:horizons[-1]] = traj[:, self.reward_index] * self.gamma ** np.arange(horizons[-1])
            traj_returns = np.vstack([traj_returns, [traj_return]])

            # Compute the trajectory log policy gradient
            traj_log_gradient = np.zeros((self.mdp.horizon, self.dim))
            traj_log_gradient[0] = self.policy.gradient_log(traj[0, self.state_index], traj[0, self.action_index])
            for i in range(1, horizons[-1]):
                traj_log_gradient[i] = traj_log_gradient[i-1] + self.policy.gradient_log(traj[i, self.state_index], traj[i, self.action_index])

            traj_log_gradients = np.vstack([traj_log_gradients, [traj_log_gradient]])


            # Compute the optimal baseline
            if baseline_type == 'vectorial':
                baseline = np.sum(traj_log_gradients ** 2 * traj_returns[:, :,  np.newaxis], axis=0) \
                        / (self.eps + np.sum(traj_log_gradients ** 2, axis=0))[np.newaxis]

            elif baseline_type == 'scalar':
                baseline = np.sum(la.norm(traj_log_gradients ** 2 * traj_returns[:, :, np.newaxis], axis=2), axis=0) \
                           / (self.eps + np.sum(la.norm(traj_log_gradients ** 2, axis=2), axis=0))[np.newaxis]
            else:
                baseline = 0.

            # Compute the gradient estimate
            old_gradient_estimate = gradient_estimate
            gradient_estimate = 1. / ite * np.sum(traj_log_gradients * (traj_returns[:, :, np.newaxis] - baseline), axis=(1,0))

            # Compute the gradient increment
            gradient_increment = la.norm(
                gradient_estimate - old_gradient_estimate)

            if self.verbose:
                print('\tIteration %s return %s gradient_norm %s gradient_increment %s' % (ite, traj_return, la.norm(gradient_estimate), gradient_increment))

        return gradient_estimate, np.mean(np.sum(traj_returns, axis=1))

