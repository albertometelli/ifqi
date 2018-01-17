from ifqi.algorithms.policy_gradient.policy import GaussianPolicyLinearMean, \
    DeterministicPolicyLinearMean, GaussianPolicyLinearMeanCholeskyVar
from ifqi.evaluation.evaluation import collect_episodes
from ifqi.evaluation.trajectory_generator import OnlineTrajectoryGenerator, \
    OfflineTrajectoryGenerator
from ifqi.algorithms.policy_gradient.policy_gradient_learner import \
    PolicyGradientLearner
import numpy as np


class OnOffLearner:

    def __init__(self,
                 mdp,
                initial_mu,
                initial_sigma,
                learn_sigma = True,
                initial_batch_size=200,
                batch_size_incr=100,
                max_batch_size=5000,
                select_initial_point=False,
                adaptive_stop=True,
                safe_stopping=False,
                search_horizon=False,
                adapt_batchsize=False,
                bound='chebyshev',
                delta=0.2,
                importance_weighting_method='is',
                learning_rate=0.002,
                estimator='gpomdp',
                gradient_updater='vanilla',
                max_offline_iterations=50,
                online_iterations=100,
                verbose=1):
        self.mdp = mdp
        self.initial_mu = initial_mu
        self.initial_sigma = initial_sigma
        self.learn_sigma = learn_sigma
        self.initial_batch_size=initial_batch_size
        self.batch_size_incr = batch_size_incr
        self.max_batch_size = max_batch_size
        self.select_initial_point = select_initial_point
        self.adaptive_stop = adaptive_stop
        self.safe_stopping = safe_stopping
        self.search_horizon = search_horizon
        self.adapt_batchsize = adapt_batchsize
        self.bound = bound
        self.delta = delta
        self.importance_weighting_method = importance_weighting_method
        self.learning_rate = learning_rate
        self.estimator = estimator
        self.gradient_updater = gradient_updater
        self.max_offline_iterations = max_offline_iterations
        self.online_iterations = online_iterations
        self.verbose = verbose

    def learn(self):
        if self.verbose: print("START ONLINE/OFFLINE LEARNING")
        N = self.initial_batch_size

        if self.learn_sigma:
            behavioral_policy = GaussianPolicyLinearMeanCholeskyVar(self.initial_mu,
                                                                self.initial_sigma)
            target_policy = GaussianPolicyLinearMeanCholeskyVar(self.initial_mu,
                                                            self.initial_sigma)
        else:
            behavioral_policy = GaussianPolicyLinearMean(self.initial_mu,
                                                            self.initial_sigma)
            target_policy =   GaussianPolicyLinearMean(self.initial_mu,
                                                            self.initial_sigma)

        history = []
        offline_history_lens = [0]
        dataset = collect_episodes(self.mdp, behavioral_policy,
                                   n_episodes=N)

        if self.verbose: print("\nSTART EPOCH 0 with dataset of size %s" % (N))
        i = 0
        while i<self.online_iterations:
            offline_trajectory_generator = OfflineTrajectoryGenerator(dataset)

            offline_learner = PolicyGradientLearner(offline_trajectory_generator,
                                           target_policy,
                                           self.mdp.gamma,
                                           self.mdp.horizon,
                                           select_initial_point = self.select_initial_point,
                                           select_optimal_horizon=False,
                                           adaptive_stop=self.adaptive_stop,
                                           safe_stopping = self.safe_stopping,
                                           hill_climb = self.search_horizon,
                                           bound=self.bound,
                                           delta=self.delta,
                                           behavioral_policy=behavioral_policy,
                                           importance_weighting_method=self.importance_weighting_method,
                                           learning_rate=self.learning_rate,
                                           estimator=self.estimator,
                                           gradient_updater=self.gradient_updater,
                                           max_iter_opt=self.max_offline_iterations,
                                           max_iter_eval=N,
                                           verbose=self.verbose - 1)

            initial_parameter = behavioral_policy.get_parameter()

            optimal_parameter, offline_history = offline_learner.optimize(initial_parameter,
                                                                          return_history=True)

            offline_history_lens.append(len(offline_history)-1)
            history.extend(offline_history[:-1])
            behavioral_policy.set_parameter(optimal_parameter)
            target_policy.set_parameter(optimal_parameter)

            actual_iterations = len(offline_history) - 1
            if self.adapt_batchsize and actual_iterations == 0:
                if N+self.batch_size_incr>self.max_batch_size:
                    if self.verbose: print('Cannot learn further with the available data!')
                    break

                N+=self.batch_size_incr
                print('Collecting %s more trajectories (total: %s)' % \
                      (self.batch_size_incr, N))
                dataset = np.concatenate((dataset,collect_episodes(self.mdp, behavioral_policy,
                                                   n_episodes=self.batch_size_incr)))
            else:
                if self.verbose:
                    print('******** RECAP OF EPOCH %s: ********' % (i))
                    print('Initial parameter: %s' % initial_parameter)
                    print('Optimal parameter: %s' % optimal_parameter)
                    print('Iterations: %s' % (len(offline_history) - 1))
                    print('************************************')

                    i+=1
                    print("\nSTART EPOCH %s with new dataset of size %s" % \
                          (i, N))

                dataset = collect_episodes(self.mdp, behavioral_policy, n_episodes=N)

        if self.verbose: print('\nEND ONLINE/OFFLINE LEARNING')


        history.append(offline_history[-1])
        history_filter = np.cumsum(offline_history_lens)

        return optimal_parameter, history, history_filter
