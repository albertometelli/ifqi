import numpy as np
import math
from  ifqi.algorithms.policy_gradient.policy_gradient_learner import *
PolicyGradientLearner
from ifqi.evaluation.trajectory_generator import OfflineTrajectoryGenerator

#Prototypes
def computeOptimalHorizon(M_infty,gamma,N,delta):
    return 8

def computeMInfty(behavioral_policy,target_policy):
    return 50

def maxMInfty(H_min,gamma,N,delta):
    return 100


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
                 min_learning_rate=1e-10,
                 initial_parameter=None,
                 return_history=False,
                 max_iter=100,
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
            return (None,[]) if return_history else None

        #Compute M_infty constraint
        M_max = maxMInfty(self.H_min,self.gamma,self.N,self.delta)

        #Optimize target policy
        trajectory_generator = OfflineTrajectoryGenerator(self.dataset)
        H = min(self.H_max,math.floor(H_star))
        alpha = initial_learning_rate
        if return_history:
            history = []
        it = 1
        if verbose: print("Start offline optimization")
        while it<=max_iter and H>0:
            #Perform one step of policy gradient optimization
            theta_old = self.target_policy.get_parameter()
            pg_learner = PolicyGradientLearner(trajectory_generator,
                                               self.target_policy,
                                               self.gamma,
                                               H,
                                               learning_rate = alpha,
                                               behavioral_policy =
                                                self.behavioral_policy,
                                               importance_weighting_method='pdis',
                                               max_iter_opt = 1)
            result = pg_learner.optimize(theta_old,return_history)
            if return_history:
                theta_new = result[0]
                history = history + result[1]
            else:
                theta_new = result
            self.target_policy.set_parameter(theta_new)

            #Check M_infty constraint
            M_infty = computeMInfty(self.behavioral_policy,self.target_policy)
            if M_infty>M_max:
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




