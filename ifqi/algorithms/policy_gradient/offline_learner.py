import numpy as np
import math
from  ifqi.algorithms.policy_gradient.policy_gradient_learner import *
PolicyGradientLearner
from ifqi.evaluation.trajectory_generator import OfflineTrajectoryGenerator
import scipy
from scipy import optimize


eps = 1e-7

class OfflineLearner(object):
    """
    Abstract class
    Performs policy optimization from a fixed set of episodes (abstract
    class)
    """

    def __init__(self,
                 H_min,
                 H_max,
                 dataset,
                 behavioral_policy,
                 target_policy,
                 gamma=0.99,
                 delta=0.01,
                 batch_size = None,
                 select_initial_point = True):
        """
        Constructor
        param H_min: the minimum acceptable horizon for the task to be solved
        param H_max: max available horizon length
        param dataset: collection of episodes of the task to be solved
        param behavioral_policy: policy used to sample trajectories from the task
        param target_policy: policy to optimize
        param gamma: discount factor
        param delta: tolerance
        param batch_size: number of trajectories used to estimate the gradient;
            default behavior is full gradient
        """

        self.H_min = H_min
        self.H_max = H_max
        self.dataset = dataset
        self.behavioral_policy = behavioral_policy
        self.target_policy = target_policy
        self.gamma = gamma
        self.delta = delta

        self.trajectory_generator = OfflineTrajectoryGenerator(self.dataset)
        self.N = self.trajectory_generator.n_trajectories
        self.batch_size = batch_size if batch_size is not None else self.N
        self.select_initial_point = select_initial_point

    def computeOptimalHorizon(self,M_infty=None,M_2=None):
        raise NotImplementedError

    def withinConstraint(self):
        raise NotImplementedError

    def optimize(self,
                 initial_learning_rate,
                 learning_rate_search=False,
                 min_learning_rate=1e-6,
                 initial_parameter=None,
                 return_history=False,
                 max_iter=1000,
                 verbose=1):
        """
        Performs optimization
        param initial_learning_rate: starting point of learning rate search
        param learning_rate_search: whether to perform learning rate search or
            just use the initial learning rate
        param min_learning_rate: minimum acceptable learning_rate
        param initial_parameter: starting parameter of the target policy, by
            default copied from the behavioral_policy
        return_history: wether to return a list of tripels (parametr,average
            return, gradient) for each iteration
        param max_iter: maximum number of target policy updates
        param verbose: level of verbosity
        """
        #Set initial parameter
        if initial_parameter==None:
            theta_0 = np.copy(self.behavioral_policy.get_parameter())
        else:
            theta_0 = np.copy(initial_parameter)
        self.target_policy.set_parameter(theta_0)

        #Check if dataset is large enough 
        M_infty = self.target_policy.M_inf(self.behavioral_policy)
        H_star = self.computeOptimalHorizon(M_infty)
        if H_star<self.H_min:
            if verbose: print("Not enough data!")
            if verbose: print("Cannot optimize beyond",H_star,"steps")
            return (None,[]) if return_history else None

        #Optimize target policy
        H = min(self.H_max,math.floor(H_star))
        alpha = initial_learning_rate
        if return_history:
            history = []
        it = 1
        if verbose: print("Start offline optimization")
        while it<=max_iter and H>0:
            #Perform one step of policy gradient optimization
            theta_old = self.target_policy.get_parameter()
            if verbose: print("\n",it,": H_star =",math.floor(H_star),", theta =",theta_old,", alpha =",alpha)
            pg_learner = PolicyGradientLearner(self.trajectory_generator,
                                               self.target_policy,
                                               self.gamma,
                                               H,
                                               select_initial_point=self.select_initial_point,
                                               learning_rate = alpha,
                                               behavioral_policy =
                                                self.behavioral_policy,
                                               importance_weighting_method='pdis',
                                               max_iter_opt = 1,
                                               max_iter_eval = self.batch_size,
                                               verbose = verbose-1)
            result = pg_learner.optimize(theta_old,return_history)
            if return_history:
                theta_new = result[0]
            else:
                theta_new = result
            self.target_policy.set_parameter(theta_new)

            #Check constraint
            if not self.withinConstraint():
                #Rollback
                self.target_policy.set_parameter(theta_old)
                if learning_rate_search and alpha>min_learning_rate:
                    alpha/=2
                    continue
                else:
                    if verbose: print("Boundary reached")
                    break
            else:
                if return_history: history.append(result[1][1])
                #Compute new optimal horizon
                H_star = self.computeOptimalHorizon(M_infty)
                H = min(self.H_max,math.floor(H_star))
                if H<self.H_min: #This should not happen according to theory
                    print("UNEXPECTED: H*<H_MIN")
                    H = self.H_min
                it+=1
        if verbose: print("End optimization")

        #Return optimal parameter
        if return_history:
            return self.target_policy.get_parameter(), history
        else:
            return self.target_policy.get_parameter()


class HoeffdingOfflineLearner(OfflineLearner):
    def computeOptimalHorizon(self,M_infty=None,M_2=None):
        if M_infty is None:
            raise ValueError('Need M_infty to compute Hoeffding optimal \
                             horizon')

        return 1. / np.log(M_infty) * (np.log((self.gamma * M_infty - 1) /
                                              (np.log(self.gamma * M_infty))) +
                           np.log(np.log(self.gamma) / (self.gamma - 1)) +
                           0.5 * (np.log(2 * self.batch_size) -
                                  np.log(np.log(1. / self.delta))))

    def maxMInfty(self):
        if not hasattr(self,'_M_max'):
            f = lambda minf: self.computeOptimalHorizon(M_infty=minf) - self.H_min
            self._M_max = scipy.optimize.fsolve(f,2)
        return self._M_max

    def withinConstraint(self):
        M_infty = self.target_policy.M_inf(self.behavioral_policy)
        condition =  M_infty<=self.maxMInfty() + eps 
        if not condition:
            print("too far!",M_infty,">",self.maxMInfty())

        return condition

class ChebyshevOfflineLearner(OfflineLearner):
    def computeOptimalHorizon(self,M_infty):
        pass

