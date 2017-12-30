import numpy as np
import math
from  ifqi.algorithms.policy_gradient.policy_gradient_learner import *
PolicyGradientLearner
from ifqi.evaluation.trajectory_generator import OfflineTrajectoryGenerator
import scipy
from scipy import optimize

eps = 1e-7

#Prototypes
def computeOptimalHorizon(M_infty,gamma,N,delta):
    return int(math.floor((1 / np.log(M_infty)) *
                      (np.log((gamma * M_infty - 1) / (np.log(gamma * M_infty))) +
                       np.log(np.log(gamma) / (gamma - 1)) +
                       0.5 * (np.log(2 * N) - np.log(np.log(1 / delta))))))

def computeMInfty(behavioral_policy, target_policy):
    return ((behavioral_policy.covar / target_policy.covar) *
            np.exp(0.5 * (target_policy.get_parameter() - behavioral_policy.get_parameter())**2 * 8 ** 2 /
                     (behavioral_policy.covar ** 2 - target_policy.covar ** 2)))

def maxMInfty(H_min,gamma,N,delta):
    f = lambda minf: (gamma**H_min)/(1-gamma) - \
        (1 - (gamma*minf)**H_min)/(1 - gamma*minf) * \
        np.sqrt(np.log(1/delta) / (2*N))
    return scipy.optimize.fsolve(f, 1)


class OfflineLearner(object):
    """
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
                 delta=0.01):
        """
        Constructor
        param H_min: the minimum acceptable horizon for the task to be solved
        param H_max: max available horizon length
        param dataset: collection of episodes of the task to be solved
        param behavioral_policy: policy used to sample trajectories from the task
        param target_policy: policy to optimize
        param gamma: discount factor
        param delta: tolerance
        """

        self.H_min = H_min
        self.H_max = H_max
        self.dataset = dataset
        self.N = np.shape(dataset)[0]
        self.behavioral_policy = behavioral_policy
        self.target_policy = target_policy
        self.gamma = gamma
        self.delta = delta

    def optimize(self):
        pass


class HoeffdingOfflineLearner(OfflineLearner):

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
        M_infty = computeMInfty(self.behavioral_policy,self.target_policy)
        H_star = computeOptimalHorizon(M_infty,
                                       self.gamma,
                                       self.N,
                                       self.delta)
        if H_star<self.H_min:
            if verbose: print("Not enough data!")
            if verbose: print("Cannot optimize beyond",H_star,"steps")
            return (None,[]) if return_history else None

        #Compute M_infty constraint
        M_max = maxMInfty(self.H_min,self.gamma,self.N,self.delta)

        #Optimize target policy
        trajectory_generator = OfflineTrajectoryGenerator(self.dataset)
        H = min(self.H_max,H_star)
        alpha = initial_learning_rate
        if return_history:
            history = []
        it = 1
        if verbose: print("Start offline optimization")
        while it<=max_iter and H>0:
            #Perform one step of policy gradient optimization
            theta_old = self.target_policy.get_parameter()
            if verbose: print(it,": H_star =",H_star,", theta =",theta_old,", alpha =",alpha)
            pg_learner = PolicyGradientLearner(trajectory_generator,
                                               self.target_policy,
                                               self.gamma,
                                               H,
                                               learning_rate = alpha,
                                               behavioral_policy =
                                                self.behavioral_policy,
                                               importance_weighting_method='pdis',
                                               max_iter_opt = 1,
                                               verbose = 0)
            result = pg_learner.optimize(theta_old,return_history)
            if return_history:
                theta_new = result[0]
                history = history + result[1]
            else:
                theta_new = result
            self.target_policy.set_parameter(theta_new)

            #Check M_infty constraint
            M_infty = computeMInfty(self.behavioral_policy,self.target_policy)
            if M_infty>M_max + eps:
                if verbose: print(theta_new,"is too far!",M_infty,">",M_max)
                #Rollback
                self.target_policy.set_parameter(theta_old)
                if learning_rate_search and alpha>min_learning_rate:
                    alpha/=2
                    continue
                else:
                    if verbose: print("Boundary reached")
                    break
            else:
                #Compute new optimal horizon
                H_star = computeOptimalHorizon(M_infty,self.gamma,self.N,self.delta) 
                H = min(self.H_max,math.floor(H_star))
                it+=1
        if verbose: print("End optimization")

        #Return optimal parameter
        if return_history:
            return self.target_policy.get_parameter(), history
        else:
            return self.target_policy.get_parameter()




