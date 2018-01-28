from ifqi.algorithms.policy_gradient.policy import GaussianPolicyLinearMean, \
    DeterministicPolicyLinearMean, GaussianPolicyLinearMeanCholeskyVar
from ifqi.evaluation.evaluation import collect_episodes
from ifqi.evaluation.trajectory_generator import OnlineTrajectoryGenerator, \
    OfflineTrajectoryGenerator
from ifqi.algorithms.policy_gradient.policy_gradient_learner import \
    PolicyGradientLearner
import numpy as np
from ifqi.algorithms.policy_gradient.gradient_descent import *
from tabulate import tabulate
import csv


class OnOffLearner:

    def __init__(self,
                 mdp,
                behavioral_policy,
                target_policy,
                initial_batch_size=200,
                batch_size_incr=100,
                max_batch_size=5000,
                select_initial_point=False,
                select_optimal_horizon=False,
                adaptive_stop=True,
                safe_stopping=False,
                search_horizon=False,
                adapt_batchsize=False,
                optimize_bound=False,
                 search_step_size=False,
                bound='chebyshev',
                delta=0.2,
                importance_weighting_method='is',
                learning_rate=0.002,
                estimator='gpomdp',
                gradient_updater='vanilla',
                gradient_updater_outer='vanilla',
                max_offline_iterations=50,
                online_iterations=100,
                state_index=0,
                action_index=1,
                reward_index=2,
                verbose=1,
                file_online_epochs=None,
                 file_offline_epochs=None,
                 return_history=1,
                parallelize=False,
                 natural=False,
                 normalize_return=False):
        self.mdp = mdp
        self.behavioral_policy = behavioral_policy
        self.target_policy = target_policy
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
        self.optimize_bound = optimize_bound
        self.gradient_updater_outer = gradient_updater_outer
        self.state_index = state_index
        self.reward_index = reward_index
        self.action_index = action_index
        self.select_optimal_horizon = select_optimal_horizon
        self.search_step_size = search_step_size
        self.file_online_epochs = file_online_epochs
        self.file_offline_epochs = file_offline_epochs
        self.return_history = return_history
        self.parallelize = parallelize
        self.natural = natural
        self.normalize_return = normalize_return

        if gradient_updater_outer == 'vanilla':
            self.gradient_updater_outer = VanillaGradient(self.learning_rate, ascent=True)
        elif gradient_updater_outer == 'adam':
            self.gradient_updater_outer = Adam(self.learning_rate, ascent=True)
        elif gradient_updater_outer == 'annelling':
            self.gradient_updater_outer = AnnellingGradient(self.learning_rate, ascent=True)
        elif gradient_updater_outer == 'rmsprop':
            self.gradient_updater_outer = RMSProp(self.learning_rate, ascent=True)
        elif gradient_updater_outer == 'adagrad':
            self.gradient_updater_outer = AdaGrad(self.learning_rate, ascent=True)
        else:
            raise ValueError('Gradient updater not found.')

        self.gradient_updater_outer.initialize(0)

    def learn(self):
        if self.verbose: print("START ONLINE/OFFLINE LEARNING")
        N = self.initial_batch_size

        history = []
        offline_history_lens = [0]
        dataset = collect_episodes(self.mdp, self.behavioral_policy,
                                   n_episodes=N)

        if self.file_online_epochs is not None:
            file_online_epochs_desc = open(self.file_online_epochs, 'w')
            file_online_epochs_writer = csv.writer(file_online_epochs_desc, delimiter=',')
        if self.file_offline_epochs is not None:
            file_offline_epochs_desc = open(self.file_offline_epochs, 'w')
            file_offline_epochs_writer = csv.writer(file_offline_epochs_desc, delimiter=',')

        if self.verbose:
            print("\nSTART EPOCH 0 with dataset of size %s" % (N))

        i = 0

        while i < self.online_iterations:
            offline_trajectory_generator = OfflineTrajectoryGenerator(dataset)
 
            offline_learner = PolicyGradientLearner(offline_trajectory_generator,
                                                    self.target_policy,
                                           self.mdp.gamma,
                                           self.mdp.horizon,
                                           select_initial_point = self.select_initial_point,
                                           select_optimal_horizon=self.select_optimal_horizon,
                                           adaptive_stop=self.adaptive_stop,
                                           safe_stopping = self.safe_stopping,
                                           hill_climb = self.search_horizon,
                                           bound=self.bound,
                                           optimize_bound=self.optimize_bound,
                                           search_step_size=self.search_step_size,
                                           delta=self.delta,
                                           behavioral_policy=self.behavioral_policy,
                                           importance_weighting_method=self.importance_weighting_method,
                                           learning_rate=self.learning_rate,
                                           estimator=self.estimator,
                                           gradient_updater=self.gradient_updater,
                                           max_iter_opt=self.max_offline_iterations,
                                           max_iter_eval=N,
                                           verbose=self.verbose - 1,
                                           state_index=self.state_index,
                                           action_index=self.action_index,
                                           reward_index=self.reward_index,
                                           max_reward=self.mdp.max_reward,
                                           min_reward=self.mdp.min_reward,
                                           parallelize=self.parallelize,
                                                    natural=self.natural,
                                                    normalize_return=self.normalize_return)


            gradient, _, _, _, _, _, _, _, _ = offline_learner.estimator.estimate()
            self.learning_rate = self.gradient_updater_outer.get_learning_rate(gradient)
            offline_learner.gradient_updater.learning_rate = self.learning_rate

            initial_parameter = self.behavioral_policy.get_parameter() 
            optimal_parameter, offline_history = offline_learner.optimize(initial_parameter,
                                                                          return_history=self.return_history) 
            offline_iterations = len(offline_history) - 1
            offline_history_lens.append(offline_iterations)
            history.extend(offline_history[:-1])

            self.behavioral_policy.set_parameter(optimal_parameter)
            self.target_policy.set_parameter(optimal_parameter)


            if self.adapt_batchsize and offline_iterations == 0:
                if N + self.batch_size_incr > self.max_batch_size:
                    if self.verbose:
                        print('ADAPTIVE BATCH SIZE - Reached maximum batch size')
                    break

                N += self.batch_size_incr
                print('ADAPTIVE BATCH SIZE - Collecting %s more trajectories (total: %s)' % \
                      (self.batch_size_incr, N))
                dataset = np.concatenate((dataset,collect_episodes(self.mdp, self.behavioral_policy,
                                                   n_episodes=self.batch_size_incr)))
            else:

                if self.verbose:
                    print(tabulate([('Epoch', i),
                                    ('InitialParameter', initial_parameter),
                                    ('FinalParameter', optimal_parameter),
                                    ('OfflineIterations', offline_iterations),
                                    ('BatchSize', N),
                                    ('InitialStepSize', self.learning_rate)], tablefmt="rst"))

                    print("\nSTART EPOCH %s with new dataset of size %s" % \
                          (i, N))

                if self.file_offline_epochs is not None:
                    if i == 0:
                        file_offline_epochs_writer.writerow(['Epoch,' + offline_learner.csv_header])
                    for j in range(offline_iterations):
                        file_offline_epochs_writer.writerow([i + 1] + offline_history[j])

                if self.file_online_epochs is not None:
                    if i == 0:
                        file_online_epochs_writer.writerow(['Epoch,InitialParameter,FinalParameter,OfflineIterations,BatchSize,InitialStepSize'])
                    file_online_epochs_writer.writerow([i, initial_parameter, optimal_parameter, offline_iterations, N, self.learning_rate])

                N = self.initial_batch_size
                dataset = collect_episodes(self.mdp, self.behavioral_policy, n_episodes=N)
                i += 1

        if self.file_online_epochs:
            file_online_epochs_desc.close()

        if self.file_offline_epochs:
            file_offline_epochs_desc.close()

        if self.verbose:
            print('\nEND ONLINE/OFFLINE LEARNING')


        history.append(offline_history[-1])
        history_filter = np.cumsum(offline_history_lens)

        #file_complete.close()
        #file_online.close()

        return optimal_parameter, history, history_filter
