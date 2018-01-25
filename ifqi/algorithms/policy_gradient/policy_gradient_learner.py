import numpy.linalg as la
import numpy as np
from ifqi.evaluation import evaluation
import sys
import copy
from ifqi.algorithms.policy_gradient.gradient_descent import *
from ifqi.algorithms.importance_weighting.importance_weighting import *
from ifqi.algorithms.bound.bound import *
import ifqi.algorithms.bound.bound_factory as bound_factory
from tabulate import tabulate
from ifqi.utils.tictoc import *
from joblib import Parallel, delayed

eps = 1e-12

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
                 adaptive_stop=False,
                 safe_stopping=False,
                 hill_climb = False,
                 optimize_bound = False,
                 search_step_size=False,
                 max_iter_eval=100,
                 tol_eval=-1.,
                 max_iter_opt=100,
                 tol_opt=-1.,
                 verbose=0,
                 state_index=0,
                 action_index=1,
                 reward_index=2,
                 max_reward=1.,
                 min_reward=0.,
                 parallelize=True,
                 natural=False):

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
        self.behavioral_policy = behavioral_policy.get_copy()if \
                behavioral_policy is not None else None
        self.target_policy = target_policy.get_copy()
        self.gamma = gamma
        self.horizon = horizon
        self.learning_rate = learning_rate
        self.max_iter_eval = max_iter_eval
        self.tol_eval = tol_eval
        self.max_iter_opt = max_iter_opt
        self.tol_opt = tol_opt
        self.verbose = verbose
        self.state_index = state_index
        self.action_index = action_index
        self.reward_index = reward_index
        self.baseline_type = baseline_type
        self.delta = delta
        self.select_initial_point = select_initial_point
        self.select_optimal_horizon = select_optimal_horizon
        self.adaptive_stop = adaptive_stop
        self.safe_stopping = safe_stopping
        self.hill_climb = hill_climb
        self.optimize_bound = optimize_bound
        self.search_step_size = search_step_size
        self.max_reward, self.min_reward = max_reward, min_reward
        self.parallelize = parallelize
        self.natural = natural

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
        elif bound == 'normal':
            self.bound = bound_factory.build_normal_bound(self.is_estimator,
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
                                                        self.optimize_bound,
                                                        self.tol_eval,
                                                        self.max_iter_eval,
                                                        self.verbose == 2,
                                                        self.state_index,
                                                        self.action_index,
                                                        self.reward_index,
                                                        self.max_reward,
                                                        self.min_reward)
        elif estimator == 'gpomdp':
            self.estimator = GPOMDPGradientEstimator(self.trajectory_generator,
                                                    self.behavioral_policy,
                                                      self.target_policy,
                                                     self.is_estimator,
                                                     self.gamma,
                                                     self.horizon,
                                                     self.select_initial_point,
                                                     self.bound,
                                                     self.optimize_bound,
                                                     self.tol_eval,
                                                     self.max_iter_eval,
                                                     self.verbose == 2,
                                                     self.state_index,
                                                     self.action_index,
                                                     self.reward_index,
                                                     self.max_reward,
                                                     self.min_reward)
        else:
            raise ValueError('Gradient estimator not found.')

        if gradient_updater == 'vanilla':
            self.gradient_updater = VanillaGradient(self.learning_rate, ascent=True)
        elif gradient_updater == 'adam':
            self.gradient_updater = Adam(self.learning_rate, ascent=True)
        elif gradient_updater == 'annelling':
            self.gradient_updater = AnnellingGradient(self.learning_rate, ascent=True)
        elif gradient_updater == 'chebychev-adaptive':
            self.gradient_updater = ChebychevAdaptiveGradient(self.learning_rate, self.max_iter_eval, self.delta, self.gamma, self.horizon, ascent=True)
        else:
            raise ValueError('Gradient updater not found.')

        if self.parallelize:
            self._estimator_function = self.estimator.estimate_parallel
        else:
            self._estimator_function = self.estimator.estimate

    def optimize(self, initial_parameter, return_history=0):
        '''
        This method performs the optimization of the parameters of the target_policy

        param initial_parameter: the initial value of the parameter (n_parameters,)
        param return_history: whether to return a list of tripels (parameter,
                              average return, gradient) for each iteration
        '''

        if return_history > 0:
            self.csv_header = 'Iteration,AvgReturnNormalized,AvgReturn,AvgDiscountedReturn,Penalization,M_2,M_inf,BoundValue,GradientNorm,StepSize,Horizon'
            if return_history == 2:
                self.csv_header += 'Parameter,Gradient'

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

        if self.natural:
            #Natural gradient
            gradient, inv_fisher, avg_return_norm, penalization, H_star, \
                stepwise_avg_return, avg_return, avg_discounted_return = \
                    self.estimator.estimate_natural(baseline_type=self.baseline_type)
            gradient = np.dot(inv_fisher,gradient)
        else:
            #Vanilla gradient
            gradient, avg_return_norm, penalization, H_star, stepwise_avg_return, avg_return, avg_discounted_return = self._estimator_function(baseline_type=self.baseline_type)
        old_stepwise_avg_return = stepwise_avg_return
        gradient_norm = la.norm(gradient)
        initial_bound_value = avg_return_norm + penalization

        if return_history > 0:
            self.csv_header = 'Iteration,AvgReturnNormalized,AvgReturn,AvgDiscountedReturn,Penalization,M_2,M_inf,BoundValue,GradientNorm,StepSize,Horizon'
            elem = [ite + 1,
                    avg_return_norm,
                    avg_return,
                    avg_discounted_return,
                    penalization,
                    self.bound.M_2 if hasattr(self.bound, 'M_2') else '---',
                    self.bound.M_inf if hasattr(self.bound, 'M_inf') else '---',
                    initial_bound_value,
                    gradient_norm,
                    self.gradient_updater.learning_rate,
                    self.bound.H_star]
            if return_history == 2:
                self.csv_header += 'Parameter,Gradient'
                elem = elem + [np.copy(theta), np.copy(gradient)]
            history = [elem]

        if self.verbose >= 1:
            print(tabulate([('Iteration', ite),
                            ('AvgReturnNormalized', avg_return_norm),
                            ('AvgReturn', avg_return),
                            ('AvgDiscountedReturn', avg_discounted_return),
                            ('Penalization', penalization),
                            ('M_2', self.bound.M_2 if hasattr(self.bound, 'M_2') else '---'),
                            ('M_inf', self.bound.M_inf if hasattr(self.bound, 'M_inf') else '---'),
                            ('InitialBoundValue', initial_bound_value),
                            ('BoundValue', initial_bound_value),
                            ('GradientNorm', gradient_norm),
                            ('StepSize', self.gradient_updater.learning_rate),
                            ('Horizon', self.bound.H_star),
                            ('Parameter', theta),
                            ('Gradient', gradient)]))
            print('Real LR: %s' % self.gradient_updater.get_learning_rate(gradient))

        while ite < self.max_iter_opt and gradient_norm > self.tol_opt:

            theta_old = np.copy(theta) #Backup for safe stopping
            theta = self.gradient_updater.update(gradient) #Gradient ascent update

            self.target_policy.set_parameter(theta)
            self.estimator.set_target_policy(self.target_policy)

            old_stepwise_avg_return = stepwise_avg_return
            old_gradient = gradient
            old_avg_return_norm = avg_return_norm
            old_penalization = penalization
            old_H_star = H_star

            if self.natural:
                #Natural
                gradient, inv_fisher, avg_return_norm, penalization, H_star, stepwise_avg_return, avg_return, avg_discounted_return = \
                    self.estimator.estimate_natural(baseline_type=self.baseline_type)
                gradient = np.dot(inv_fisher,gradient)
            else:
                #Vanilla
                gradient, avg_return_norm, penalization, H_star, stepwise_avg_return, avg_return, avg_discounted_return = self._estimator_function(baseline_type=self.baseline_type)
            gradient_norm = la.norm(gradient)
            bound_value = avg_return_norm + penalization

            ite += 1
            if return_history > 0:
                elem = [ite + 1,
                        avg_return_norm,
                        avg_return,
                        avg_discounted_return,
                        penalization,
                        self.bound.M_2 if hasattr(self.bound, 'M_2') else '---',
                        self.bound.M_inf if hasattr(self.bound, 'M_inf') else '---',
                        initial_bound_value,
                        gradient_norm,
                        self.gradient_updater.learning_rate,
                        self.bound.H_star]
                if return_history == 2:
                    elem = elem + [np.copy(theta), np.copy(gradient)]
                history.append(elem)

            if self.verbose >= 1:
                print(tabulate([('Iteration', ite),
                                ('AvgReturnNormalized', avg_return_norm),
                                ('AvgReturn', avg_return),
                                ('AvgDiscountedReturn', avg_discounted_return),
                                ('Penalization', penalization),
                                ('M_2', self.bound.M_2 if hasattr(self.bound, 'M_2') else '---'),
                                ('M_inf', self.bound.M_inf if hasattr(self.bound, 'M_inf') else '---'),
                                ('InitialBoundValue', initial_bound_value),
                                ('BoundValue', bound_value),
                                ('GradientNorm', gradient_norm),
                                ('StepSize', self.gradient_updater.learning_rate),
                                ('Horizon', self.bound.H_star),
                                ('Parameter', theta),
                                ('Gradient', gradient)]))
                print('Real LR: %s' % self.gradient_updater.get_learning_rate(gradient))

            if self.adaptive_stop and bound_value < initial_bound_value:

                print('ADAPTIVE STOP - bound_value < initial_bound_value')

                if self.hill_climb:

                    new_horizon = self.compute_new_horizon(initial_bound_value, stepwise_avg_return)
                    if new_horizon != -1:
                        print('ADAPTIVE HORIZON - new horizon %s' % new_horizon)
                        self.bound.set_horizon(new_horizon)
                        assert(self.bound.H_star == new_horizon)
                    else:
                        print('ADAPTIVE HORIZON - no horizon found!')

                        # Stopping
                        if self.safe_stopping:

                            print('SAFE STOPPING')
                            theta = np.copy(theta_old)
                            self.target_policy.set_parameter(theta)
                            self.estimator.set_target_policy(self.target_policy)
                            self.gradient_updater.set_parameter(theta)
                            avg_return_norm, gradient, penalization, H_star = old_avg_return_norm, old_gradient, old_penalization, old_H_star
                            stepwise_avg_return = old_stepwise_avg_return
                            gradient_norm = la.norm(gradient)
                            ite -= 1

                            if return_history:
                                history = history[:-1]

                        if self.search_step_size and self.gradient_updater.learning_rate > 1e-6:
                            self.gradient_updater.reduce_learning_rate()
                            print('ADAPTIVE STEP SIZE - reducing to %s' % self.gradient_updater.learning_rate)
                        else:
                            print('Terminating OFFLINE update.')
                            break
                else:
                    print('Terminating OFFLINE update.')
                    break

        return theta, history

    def compute_new_horizon(self, initial_bound_value, stepwise_avg_return):
        if self.bound is None or isinstance(self.bound, DummyBound):
            return self.horizon

        for h in range(self.horizon):
            self.bound.set_horizon(h + 1)
            penalization = self.bound.penalization()
            if stepwise_avg_return[h] + penalization > initial_bound_value:
                self.bound.set_horizon(self.horizon)
                return h + 1

        self.bound.set_horizon(self.horizon)
        return -1


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
                 optimize_bound=False,
                 tol=1e-5,
                 max_iter=100,
                 verbose=True,
                 state_index=0,
                 action_index=1,
                 reward_index=2,
                 max_reward=1.,
                 min_reward=0.):
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
        self.optimize_bound = optimize_bound
        self.max_reward, self.min_reward = max_reward, min_reward

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
        traj_returns = np.zeros((self.max_iter, int(self.horizon)))

        # matrix of the tragectory log policy gradient
        traj_log_gradients = np.zeros((self.max_iter, int(self.horizon), self.dim))

        #vector of the importance weights
        ws = np.ones((self.max_iter, int(self.horizon)))

        avg_return = 0.
        avg_discounted_return = 0.

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
            H_star = self.horizon
        H_star = int(min(self.horizon, H_star))
        if self.verbose:
            print("Hstar %s" % H_star)

        if self.bound is not None:
            penalization_gradient = self.bound.gradient_penalization()
            penalization = self.bound.penalization()
        else:
            penalization_gradient = penalization = 0.

        self.trajectory_generator.set_policy(self.target_policy)

        while ite < self.max_iter:
            # Collect a trajectory
            traj = self.trajectory_generator.next()
            if not self.select_initial_point or len(traj) <= H_star:
                k = 0
            else:
                k = np.random.randint(0, len(traj) - H_star)
            traj = traj[k:k + H_star]

            #Compute lenght
            traj_horizon = traj.shape[0]
            horizons.append(traj_horizon)

            #Compute the is weights
            w = self.is_estimator.weight(traj)
            ws[ite, :traj_horizon] = w

            # Compute the trajectory return
            rewards = np.zeros(self.horizon)
            rewards[:traj_horizon] = traj[:, self.reward_index]

            avg_return += np.sum(ws[ite] * rewards)
            avg_discounted_return += np.sum(ws[ite] * self.gamma ** k * rewards * self.gamma ** np.arange(self.horizon))

            # Normalize the reward
            rewards_norm = (rewards - self.min_reward) / (self.max_reward - self.min_reward)

            traj_return = self.gamma ** k * rewards_norm * self.gamma ** np.arange(self.horizon)
            traj_returns[ite, :] = traj_return

            # Compute the trajectory log policy gradient
            traj_log_gradient = self.compute_gradient_log_policy_sum(traj)
            traj_log_gradients[ite, :traj_horizon] = traj_log_gradient

            ite += 1

        # Compute the optimal baseline
        baseline = self.compute_baseline(baseline_type, traj_log_gradients, ws, traj_returns)

        # Compute the gradient estimate
        gradient_estimate = np.mean(np.sum(traj_log_gradients * \
                                ws[:, :, np.newaxis] * \
                                (traj_returns[:, :, np.newaxis] - baseline), axis=1), axis=0)

        if self.optimize_bound:
            raise NotImplementedError()
            gradient_estimate = gradient_estimate + self.gamma ** k * penalization_gradient


        avg_return /= self.max_iter
        avg_discounted_return /= self.max_iter

        return gradient_estimate, np.mean(np.sum(traj_returns * ws, axis=1)), penalization, H_star, np.mean(np.cumsum(traj_returns * ws, axis=1), axis=0), avg_return, avg_discounted_return

    def estimate_parallel(self, baseline_type='vectorial'):

        if self.verbose:
            print('\tReinforce: starting estimation...')

        ite = 0
        gradient_estimate = np.inf

        #vector of the trajectory returns
        traj_returns = np.zeros((self.max_iter, int(self.horizon)))

        # matrix of the tragectory log policy gradient
        traj_log_gradients = np.zeros((self.max_iter, int(self.horizon), self.dim))

        #vector of the importance weights
        ws = np.ones((self.max_iter, int(self.horizon)))

        avg_return = 0.
        avg_discounted_return = 0.

        #vectors of the lenghts of trajectories
        horizons = np.zeros(self.max_iter, dtype=int)

        if self.bound is not None:
            self.bound.set_policies(self.behavioral_policy, self.target_policy)
            H_star = self.bound.H_star
            if self.verbose:
                if isinstance(self.bound, ChebyshevBound):
                    print("M 2 %s" % self.bound.M_2)
                elif isinstance(self.bound, HoeffdingBound):
                    print("M inf %s" % self.bound.M_inf)
        else:
            H_star = self.horizon
        H_star = int(min(self.horizon, H_star))
        self.H_star = H_star
        if self.verbose:
            print("Hstar %s" % H_star)

        if self.bound is not None:
            penalization_gradient = self.bound.gradient_penalization()
            penalization = self.bound.penalization()
        else:
            penalization_gradient = penalization = 0.

        self.trajectory_generator.set_policy(self.target_policy)

        result = Parallel(n_jobs=-1)(
            delayed(self.process_trajectory)(index) for index in range(self.max_iter))

        for i in range(self.max_iter):
            traj_returns[i] = result[i][0]
            horizons[i] = result[i][3]
            traj_log_gradients[i, :horizons[i]] = result[i][1]
            ws[i] = result[i][2]
            avg_return += result[i][4]
            avg_discounted_return += result[i][5]

        # Compute the optimal baseline
        baseline = self.compute_baseline(baseline_type, traj_log_gradients, ws, traj_returns)

        # Compute the gradient estimate
        gradient_estimate = np.mean(np.sum(traj_log_gradients * \
                                ws[:, :, np.newaxis] * \
                                (traj_returns[:, :, np.newaxis] - baseline), axis=1), axis=0)

        if self.optimize_bound:
            raise NotImplementedError()
            #gradient_estimate = gradient_estimate + self.gamma ** k * penalization_gradient


        assert(traj_returns.shape == ws.shape)

        avg_return /= self.max_iter
        avg_discounted_return /= self.max_iter

        return gradient_estimate, np.mean(np.sum(traj_returns * ws, axis=1)), penalization, H_star, np.mean(np.cumsum(traj_returns * ws, axis=1), axis=0), avg_return, avg_discounted_return


    def estimate_natural(self, baseline_type='vectorial'):
        if self.verbose:
            print('\tNatural gradient: starting estimation...')

        ite = 0
        gradient_estimate = np.inf

        #vector of the trajectory returns
        traj_returns = np.zeros((self.max_iter, int(self.horizon)))

        # matrix of the tragectory log policy gradient
        traj_log_gradients = np.zeros((self.max_iter, int(self.horizon), self.dim))

        #vector of the importance weights
        ws = np.ones((self.max_iter, int(self.horizon)))

        avg_return = 0.
        avg_discounted_return = 0.

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
            H_star = self.horizon
        H_star = int(min(self.horizon, H_star))
        if self.verbose:
            print("Hstar %s" % H_star)

        if self.bound is not None:
            penalization_gradient = self.bound.gradient_penalization()
            penalization = self.bound.penalization()
        else:
            penalization_gradient = penalization = 0.

        self.trajectory_generator.set_policy(self.target_policy)

        #Fisher matrix estimator
        fisher_samples = np.zeros((self.max_iter,self.dim,self.dim))

        while ite < self.max_iter:
            # Collect a trajectory
            traj = self.trajectory_generator.next()
            if not self.select_initial_point or len(traj) <= H_star:
                k = 0
            else:
                k = np.random.randint(0, len(traj) - H_star)
            traj = traj[k:k + H_star]

            #Compute lenght
            traj_horizon = traj.shape[0]
            horizons.append(traj_horizon)

            #Compute the is weights
            w = self.is_estimator.weight(traj)
            ws[ite, :traj_horizon] = w

            # Compute the trajectory return
            rewards = np.zeros(self.horizon)
            rewards[:traj_horizon] = traj[:, self.reward_index]

            avg_return += np.sum(ws[ite] * rewards)
            avg_discounted_return += np.sum(ws[ite] * self.gamma ** k * rewards * self.gamma ** np.arange(self.horizon))

            # Normalize the reward
            rewards_norm = (rewards - self.min_reward) / (self.max_reward - self.min_reward)

            traj_return = self.gamma ** k * rewards_norm * self.gamma ** np.arange(self.horizon)
            traj_returns[ite, :] = traj_return

            # Compute the trajectory log policy gradient
            traj_log_gradient = self.compute_gradient_log_policy_sum(traj)
            traj_log_gradients[ite, :traj_horizon] = traj_log_gradient

            #Compute Fisher sample TODO: optimize over gradlog sum
            #Use per-trajectory importance weights
            fisher_weight = w[-1]
            fisher_samples[ite,:,:] = fisher_weight*self.compute_fisher_sample(traj)

            ite += 1

        # Compute the optimal baseline
        baseline = self.compute_baseline(baseline_type, traj_log_gradients, ws, traj_returns)

        # Compute the gradient estimate
        gradient_estimate = np.mean(np.sum(traj_log_gradients * \
                                ws[:, :, np.newaxis] * \
                                (traj_returns[:, :, np.newaxis] - baseline), axis=1), axis=0)

        if self.optimize_bound:
            raise NotImplementedError()
            gradient_estimate = gradient_estimate + self.gamma ** k * penalization_gradient

        fisher = np.mean(fisher_samples,0)
        inv_fisher = np.linalg.inv(fisher+eps)

        avg_return /= self.max_iter
        avg_discounted_return /= self.max_iter

        return gradient_estimate, inv_fisher, np.mean(np.sum(traj_returns * ws, axis=1)), penalization, H_star, np.mean(np.cumsum(traj_returns * ws, axis=1), axis=0), avg_return, avg_discounted_return


    def compute_fisher_sample(self,traj):
        gradlog_sum = np.zeros((self.dim))
        for i in range(len(traj)):
            gradlog_sum+=self.target_policy.gradient_log( \
                traj[i, self.state_index], traj[i, self.action_index])

        return np.outer(gradlog_sum,gradlog_sum)

    def process_trajectory(self, index):

        # Collect a trajectory
        traj = self.trajectory_generator.dataset[index]
        if not self.select_initial_point or len(traj) <= self.H_star:
            k = 0
        else:
            k = np.random.randint(0, len(traj) - self.H_star)
        traj = traj[k:k + self.H_star]

        # Compute lenght
        traj_horizon = traj.shape[0]

        # Compute the is weights
        w = np.ones(self.horizon)
        w[:traj_horizon] = self.is_estimator.weight(traj)

        # Compute the trajectory return
        rewards = np.zeros(self.horizon)
        rewards[:traj_horizon] = traj[:, self.reward_index]

        _return = np.sum(w * rewards)
        discounted_return = np.sum(w * self.gamma ** k * rewards * self.gamma ** np.arange(self.horizon))

        # Normalize the reward
        rewards_norm = (rewards - self.min_reward) / (self.max_reward - self.min_reward)

        traj_return = self.gamma ** k * rewards_norm * self.gamma ** np.arange(self.horizon)

        # Compute the trajectory log policy gradient
        traj_log_gradient = self.compute_gradient_log_policy_sum(traj)

        return traj_return, traj_log_gradient, w, traj_horizon, _return, discounted_return

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

        return np.repeat([traj_log_gradient], horizon, axis=0)

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
        traj_log_gradient = np.zeros((horizon, self.dim))
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
