import numpy.linalg as la
import numpy as np
from ifqi.evaluation import evaluation
import sys
import copy
from ifqi.algorithms.policy_gradient.gradient_descent import *
from ifqi.algorithms.importance_weighting.importance_weighting import *
from ifqi.algorithms.bound.bound import *
import ifqi.algorithms.bound.bound_factory as bound_factory

class PolicyGradientLearner(object):

    '''
    This class performs policy search exploiting the policy gradient.
    '''

    def __init__(self,
                 trajectory_generator,
                 target_policy,
                 gamma,
                 horizon,
                 bound=None,
                 delta=0.01,
                 learning_rate=0.01,
                 estimator='reinforce',
                 baseline_type='vectorial',
                 gradient_updater='vanilla',
                 behavioral_policy=None,
                 importance_weighting_method=None,
                 select_initial_point=None,
                 select_optimal_horizon=False,
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

        param trajectory_generator: a generator for trajectories either online
                                    or offline
        param target_policy: the policy whose parameters are optimized
        param gamma: discount factor
        param horizon: the maximum lenght of a trajectory
        param learning_rate: the learning rate
        param estimator: the name of the gradient estimtor to use, currently
                         only 'reinforce' and 'gpomdp' are supported
        param baseline_type: the type of baseline to be used in the gradient
                             estimate (None, 'scalar', 'vectorial')
        param gradient_updater: the method to be used to update the gradient
                                ('vanilla', 'annelling', 'adam')
        param behavioral_policy: the policy used to collect the data (only
                                 for offline estimation)
        param importance_weighting_method: the importance sampling strategy
                                           to be used (only offline), values:
                                           None, 'is', 'pdis'
        param max_iter_eval: the maximum number of iteration for gradient
                             estimation
        param tol_eval: the estimation stops when norm-2 of the gradient
                        increment is below tol_eval (negative ignored)
        param max_iter_opt: the maximum number of iteration for gradient
                             optimization
        param tol_opt: the optimization stops when norm-2 of the gradient
                       increment is below tol_eval (negative ignored)
        param verbose: verbosity level
        param state_index: a list of indeces corresponding to the positions
                           of the state variables in the trajectory samples
        param action_index: a list of indeces corresponding to the positions
                           of the action variables in the trajectory samples
        param reward_index: an index corresponding to the positions
                           of the reward variable in the trajectory samples
        '''

        self.trajectory_generator = trajectory_generator
        self.behavioral_policy = copy.deepcopy(behavioral_policy)
        self.target_policy = copy.deepcopy(target_policy)
        self.gamma = gamma
        self.horizon = horizon
        self.learning_rate = learning_rate
        self.max_iter_eval = max_iter_eval
        self.tol_eval = tol_eval
        self.max_iter_opt = max_iter_opt
        self.tol_opt = tol_opt
        self.verbose = verbose
        self.state_index= state_index
        self.action_index = action_index
        self.reward_index = reward_index
        self.baseline_type = baseline_type
        self.delta = delta
        self.select_initial_point = select_initial_point
        self.select_optimal_horizon = select_optimal_horizon

        if importance_weighting_method is not None and behavioral_policy is None:
            raise ValueError('If you want to use importance weighting you must \
                             provide a behavioral policy')

        if importance_weighting_method is None:
            self.is_estimator = DummyImportanceWeighting(
                self.behavioral_policy, self.target_policy, self.state_index, self.action_index)
        elif importance_weighting_method == 'pdis':
            self.is_estimator = PerDecisionRatioImportanceWeighting(
                self.behavioral_policy, self.target_policy, self.state_index, self.action_index)
        elif importance_weighting_method == 'is':
            self.is_estimator = RatioImportanceWeighting(
                self.behavioral_policy, self.target_policy, self.state_index, self.action_index)
        else:
            raise ValueError('Importance weighting method not found.')

        print(bound)
        if bound is None:
            self.bound = DummyBound(self.max_iter_eval,
                                    self.delta,
                                    self.gamma,
                                    self.behavioral_policy,
                                    self.target_policy,
                                    self.horizon,
                                    self.select_optimal_horizon)
        elif bound == 'hoeffding':
            self.bound = bound_factory.build_Hoeffding_bound(self.is_estimator,
                                                             self.max_iter_eval,
                                                             self.delta,
                                                             self.gamma,
                                                             self.behavioral_policy,
                                                             self.target_policy,
                                                             self.horizon,
                                                             self.select_optimal_horizon)
        elif bound == 'chebyshev':
            self.bound = bound_factory.build_Chebyshev_bound(self.is_estimator,
                                                             self.max_iter_eval,
                                                             self.delta,
                                                             self.gamma,
                                                             self.behavioral_policy,
                                                             self.target_policy,
                                                             self.horizon,
                                                             self.select_optimal_horizon)
        elif bound == 'bernstein':
            self.bound = bound_factory.build_Bernstein_bound(self.is_estimator,
                                                             self.max_iter_eval,
                                                             self.delta,
                                                             self.gamma,
                                                             self.behavioral_policy,
                                                             self.target_policy,
                                                             self.horizon,
                                                             self.select_optimal_horizon)
        else:
            raise NotImplementedError()

        if estimator == 'reinforce':
            self.estimator = ReinforceGradientEstimator(self.trajectory_generator,
                                                        self.behavioral_policy,
                                                        self.target_policy,
                                                        self.is_estimator,
                                                        self.gamma,
                                                        self.horizon,
                                                        self.select_initial_point,
                                                        self.bound,
                                                        self.tol_eval,
                                                        self.max_iter_eval,
                                                        self.verbose == 2,
                                                        self.state_index,
                                                        self.action_index,
                                                        self.reward_index)
        elif estimator == 'gpomdp':
            self.estimator = GPOMDPGradientEstimator(self.trajectory_generator,
                                                    self.behavioral_policy,
                                                      self.target_policy,
                                                     self.is_estimator,
                                                     self.gamma,
                                                     self.horizon,
                                                     self.select_initial_point,
                                                     self.bound,
                                                     self.tol_eval,
                                                     self.max_iter_eval,
                                                     self.verbose == 2,
                                                     self.state_index,
                                                     self.action_index,
                                                     self.reward_index)
        else:
            raise ValueError('Gradient estimator not found.')

        if gradient_updater == 'vanilla':
            self.gradient_updater = VanillaGradient(self.learning_rate, ascent=True)
        elif gradient_updater == 'adam':
            self.gradient_updater = Adam(self.learning_rate, ascent=True)
        elif gradient_updater == 'annelling':
            self.gradient_updater = AnnellingGradient(self.learning_rate, ascent=True)
        else:
            raise ValueError('Gradient updater not found.')

    def optimize(self, initial_parameter, return_history=False):
        '''
        This method performs the optimization of the parameters of the target_policy

        param initial_parameter: the initial value of the parameter (n_parameters,)
        param return_history: whether to return a list of tripels (parameter,
                              average return, gradient) for each iteration
        '''

        ite = 0
        theta = np.copy(initial_parameter)
        self.gradient_updater.initialize(theta)

        self.target_policy.set_parameter(theta)
        self.estimator.set_target_policy(self.target_policy)

        if self.verbose >= 1:
            print('Policy gradient: starting optimization...')
            print('Gradient estimator: %s' % self.estimator)
            print('Bound: %s' % self.bound.__class__)
            print('Trajectory generator: %s' % self.trajectory_generator.__class__)

        gradient, avg_return, penalization, H_star, terminate = self.estimator.estimate(baseline_type=self.baseline_type)

        if return_history:
            history = [[np.copy(theta), avg_return, gradient, penalization, H_star]]

        gradient_norm = la.norm(gradient)

        if self.verbose >= 1:
            print('Ite %s: return %s - gradient norm %s' % (ite, avg_return, gradient_norm))

        while ite < self.max_iter_opt and gradient_norm > self.tol_opt: #and not terminate:

            theta = self.gradient_updater.update(gradient) #Gradient ascent update
            if self.verbose >= 1:
                print(theta)

            self.target_policy.set_parameter(theta)
            self.estimator.set_target_policy(self.target_policy)
            gradient, avg_return, penalization, H_star, terminate  = self.estimator.estimate(baseline_type=self.baseline_type)

            if return_history:
                history.append([np.copy(theta), avg_return, gradient, penalization, H_star])

            #if terminate:
            #   break

            gradient_norm = la.norm(gradient)
            ite += 1

            if self.verbose >= 1:
                print('Ite %s: return %s - gradient norm %s - penalization %s' % (
                ite, avg_return, gradient_norm, penalization))
                print(terminate)

        if return_history:
            return theta, history
        else:
            return theta

class OldGradientEstimator(object):
    '''
    Abstract class for gradient estimators
    '''

    eps = 1e-24  # Tolerance used to avoid divisions by zero

    def __init__(self,
                 trajectory_generator,
                 behavioral_policy,
                 target_policy,
                 is_estimator,
                 gamma,
                 horizon,
                 tol=1e-5,
                 max_iter=100,
                 verbose=True,
                 state_index=0,
                 action_index=1,
                 reward_index=2):
        '''
        Constructor

        :param trajectory_generator: an iterable class that provides the trajectories
        :param behavioral_policy: the policy used to collect data (only offline)
        :param target_policy: the policy to be optimized
        :param is_estimator: the importance weighing estimator
        :param gamma: the discount factor
        :param horizon: the maximum lenght of a trajectory
        :param tol: the estimation stops when norm-2 of the gradient increment is
                    below tol
        :param max_iter: he maximum number of iterations for the algorithm
        :param verbose: whether to display progressing messages
        :param state_index: see PolicyGradientLearner
        :param action_index: see PolicyGradientLearner
        :param reward_index: see PolicyGradientLearner
        '''


        self.trajectory_generator = trajectory_generator
        self.behavioral_policy = behavioral_policy
        self.target_policy = target_policy
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.state_index= state_index
        self.action_index = action_index
        self.reward_index = reward_index
        self.gamma = gamma
        self.horizon = horizon
        self.is_estimator = is_estimator
        self.dim = target_policy.get_n_parameters()

    def set_target_policy(self, target_policy):
        self.target_policy = target_policy
        self.is_estimator.set_target_policy(self.target_policy)

    def compute_gradient_log_policy_sum(self, traj):
        pass

    def compute_baseline(self, baseline_type, traj_log_gradients, ws, traj_returns):
        pass

    def estimate(self, baseline_type='vectorial'):

        if self.verbose:
            print('\tReinforce: starting estimation...')

        ite = 0
        gradient_increment = np.inf
        gradient_estimate = np.inf

        #vector of the trajectory returns
        traj_returns = np.ndarray((0, int(self.horizon)))

        # matrix of the tragectory log policy gradient
        traj_log_gradients = np.ndarray((0, int(self.horizon), self.dim))

        #vector of the importance weights
        ws =  np.ndarray((0, int(self.horizon)))

        #vectors of the lenghts of trajectories
        horizons = []

        while ite < self.max_iter and gradient_increment > self.tol:
            ite += 1

            # Collect a trajectory
            self.trajectory_generator.set_policy(self.target_policy)
            traj = self.trajectory_generator.next()[:int(self.horizon)]

            #Compute lenght
            horizons.append(traj.shape[0])

            #Compute the is weights
            w = self.is_estimator.weight(traj)
            w = np.concatenate([w, np.zeros(int(self.horizon) - horizons[-1])])
            ws = np.vstack([ws, [w]])

            # Compute the trajectory return
            traj_return = np.zeros(int(self.horizon))
            traj_return[:horizons[-1]] = traj[:, self.reward_index] * self.gamma ** np.arange(horizons[-1])
            traj_returns = np.vstack([traj_returns, [traj_return]])

            # Compute the trajectory log policy gradient
            traj_log_gradient = self.compute_gradient_log_policy_sum(traj)
            traj_log_gradients = np.vstack([traj_log_gradients, [traj_log_gradient]])


            # Compute the optimal baseline
            baseline = self.compute_baseline(baseline_type, traj_log_gradients, ws,traj_returns)

            # Compute the gradient estimate
            old_gradient_estimate = gradient_estimate
            gradient_estimate = np.mean(np.sum(traj_log_gradients *  ws[:, :, np.newaxis] * (traj_returns[:, :, np.newaxis] - baseline), axis=1), axis=0)

            # Compute the gradient increment
            gradient_increment = la.norm(
                gradient_estimate - old_gradient_estimate)

            if self.verbose:
                print('\tIteration %s return %s gradient_norm %s gradient_increment %s' % (ite, traj_return, la.norm(gradient_estimate), gradient_increment))

        return gradient_estimate, np.mean(np.sum(traj_returns, axis=1))

class GradientEstimator(object):
    '''
    Abstract class for gradient estimators
    '''

    eps = 1e-24  # Tolerance used to avoid divisions by zero

    def __init__(self,
                 trajectory_generator,
                 behavioral_policy,
                 target_policy,
                 is_estimator,
                 gamma,
                 horizon,
                 select_initial_point=False,
                 bound=None,
                 tol=1e-5,
                 max_iter=100,
                 verbose=True,
                 state_index=0,
                 action_index=1,
                 reward_index=2):
        '''
        Constructor

        :param trajectory_generator: an iterable class that provides the trajectories
        :param behavioral_policy: the policy used to collect data (only offline)
        :param target_policy: the policy to be optimized
        :param is_estimator: the importance weighing estimator
        :param gamma: the discount factor
        :param horizon: the maximum lenght of a trajectory
        :param tol: the estimation stops when norm-2 of the gradient increment is
                    below tol (currently ignored)
        :param max_iter: he maximum number of iterations for the algorithm
        :param verbose: whether to display progressing messages
        :param state_index: see PolicyGradientLearner
        :param action_index: see PolicyGradientLearner
        :param reward_index: see PolicyGradientLearner
        '''


        self.trajectory_generator = trajectory_generator
        self.behavioral_policy = behavioral_policy
        self.target_policy = target_policy
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.state_index= state_index
        self.action_index = action_index
        self.reward_index = reward_index
        self.gamma = gamma
        self.horizon = horizon
        self.is_estimator = is_estimator
        self.dim = target_policy.get_n_parameters()
        self.bound = bound
        self.select_initial_point = select_initial_point

    def set_target_policy(self, target_policy):
        self.target_policy = target_policy
        self.is_estimator.set_target_policy(self.target_policy)

    def compute_gradient_log_policy_sum(self, traj):
        pass

    def compute_baseline(self, baseline_type, traj_log_gradients, ws, traj_returns):
        pass

    def estimate(self, baseline_type='vectorial'):

        if self.verbose:
            print('\tReinforce: starting estimation...')

        ite = 0
        gradient_estimate = np.inf

        #vector of the trajectory returns
        traj_returns = np.ndarray((0, int(self.horizon)))

        # matrix of the tragectory log policy gradient
        traj_log_gradients = np.ndarray((0, int(self.horizon), self.dim))

        #vector of the importance weights
        ws =  np.ndarray((0, int(self.horizon)))

        #vectors of the lenghts of trajectories
        horizons = []

        if self.bound is not None:
            self.bound.set_policies(self.behavioral_policy, self.target_policy)
            H_star = self.bound.H_star
            if self.verbose:
                if isinstance(self.bound, ChebyshevBound):
                    print("M 2 %s" % self.bound.M_2)
                elif isinstance(self.bound, HoeffdingBound):
                    print("M inf %s" % self.bound.M_inf)
        else:
            H_star = sys.maxsize
        H_star = min(int(self.horizon), H_star)
        if self.verbose:
            print("Hstar %s" % H_star)



        if self.bound is not None:
            penalization_gradient = self.bound.gradient_penalization()
            penalization = self.bound.penalization()
        else:
            penalization_gradient = penalization = 0.

        while ite < self.max_iter:
            ite += 1

            # Collect a trajectory
            self.trajectory_generator.set_policy(self.target_policy)

            traj = self.trajectory_generator.next()
            if not self.select_initial_point or len(traj) <= H_star:
                k = 0
            else:
                k = np.random.randint(0, len(traj) - H_star)
            traj = traj[k:k+H_star]

            #Compute lenght
            horizons.append(traj.shape[0])

            #Compute the is weights
            w = self.is_estimator.weight(traj)
            w = np.concatenate([w, np.zeros(int(self.horizon) - horizons[-1])])
            ws = np.vstack([ws, [w]])

            # Compute the trajectory return
            traj_return = np.zeros(int(self.horizon))
            traj_return[:horizons[-1]] = traj[:, self.reward_index] * self.gamma ** np.arange(horizons[-1])

            traj_return *= self.gamma ** k

            traj_returns = np.vstack([traj_returns, [traj_return]])

            # Compute the trajectory log policy gradient
            traj_log_gradient = self.compute_gradient_log_policy_sum(traj)
            traj_log_gradients = np.vstack([traj_log_gradients, [traj_log_gradient]])


        # Compute the optimal baseline
        baseline = self.compute_baseline(baseline_type, traj_log_gradients, ws, traj_returns)

        # Compute the gradient estimate
        gradient_estimate = np.mean(np.sum(traj_log_gradients * \
                                ws[:, :, np.newaxis] * \
                                (traj_returns[:, :, np.newaxis] - baseline), axis=1), axis=0)
        print("gradient estimate %s" % gradient_estimate)


        if la.norm(penalization_gradient) >= la.norm(gradient_estimate):
            terminate = True
        else:
            terminate = False



        gradient_estimate = gradient_estimate + self.gamma ** k * penalization_gradient


        if self.verbose:
            print("penalization gradient %s" % penalization_gradient)



        return gradient_estimate, np.mean(np.sum(traj_returns, axis=1)), penalization, H_star, terminate

class ReinforceGradientEstimator(GradientEstimator):

    '''
    This class implements the Reinforce algorithm for gradient estimation with
    element wise optimal baseline

    Williams, Ronald J. "Simple statistical gradient-following algorithms for
    connectionist reinforcement learning." Machine learning 8.3-4 (1992): 229-256.
    '''

    def compute_gradient_log_policy_sum(self, traj):
        horizon = len(traj)
        traj_log_gradient = 0.
        for i in range(0, horizon):
            traj_log_gradient += self.target_policy.gradient_log(
                traj[i, self.state_index], traj[i, self.action_index])

        return np.repeat([traj_log_gradient], self.horizon, axis=0)

    def compute_baseline(self, baseline_type, traj_log_gradients, ws, traj_returns):
        if baseline_type == 'vectorial':
            num = np.sum(traj_log_gradients ** 2 * ws[:, :, np.newaxis] ** 2 * traj_returns[:, :,  np.newaxis], axis=(0,1))
            den = np.sum(traj_log_gradients ** 2 * ws[:, :, np.newaxis] ** 2, axis=(0,1)) + 1e-24
            return (num / den)[np.newaxis, np.newaxis]

        if baseline_type == 'scalar':
            num = np.sum(la.norm(traj_log_gradients ** 2 * ws[:, :, np.newaxis] ** 2 * traj_returns[:, :, np.newaxis]), axis=(0, 1))
            den = np.sum(la.norm(traj_log_gradients ** 2 * ws[:, :, np.newaxis] ** 2), axis=(0,1)) + 1e-24
            return num / den

        return 0.



class GPOMDPGradientEstimator(GradientEstimator):
    '''
    This class implements the GPOMDP algorithm for gradient estimation with
    element wise optimal baseline

    Baxter, Jonathan, and Peter L. Bartlett. "Infinite-horizon policy-gradient
    estimation." Journal of Artificial Intelligence Research 15 (2001): 319-350.
    '''

    def compute_gradient_log_policy_sum(self, traj):
        horizon = len(traj)
        traj_log_gradient = np.zeros((self.horizon, self.dim))
        traj_log_gradient[0] = self.target_policy.gradient_log(
            traj[0, self.state_index], traj[0, self.action_index])
        for i in range(1, horizon):
            traj_log_gradient[i] = traj_log_gradient[i - 1] + self.target_policy.gradient_log(
                traj[i, self.state_index], traj[i, self.action_index])

        return traj_log_gradient

    def compute_baseline(self, baseline_type, traj_log_gradients, ws,
                         traj_returns):
        if baseline_type == 'vectorial':
            num = np.sum(traj_log_gradients ** 2 * ws[:, :, np.newaxis] ** 2 * traj_returns[:, :, np.newaxis], axis=0)
            den = np.sum(traj_log_gradients ** 2 * ws[:, :, np.newaxis] ** 2, axis=0) + 1e-24
            return (num / den)[np.newaxis]

        if baseline_type == 'scalar':
            num = np.sum(la.norm(traj_log_gradients ** 2 * ws[:, :, np.newaxis] ** 2 * traj_returns[:, :, np.newaxis]), axis=0)
            den = np.sum(la.norm(traj_log_gradients ** 2 * ws[:, :, np.newaxis] ** 2), axis=0) + 1e-24
            return (num / den)[np.newaxis, :, np.newaxis]

        return 0.

