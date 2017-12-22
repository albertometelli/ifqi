from ifqi.evaluation.evaluation import collect_episodes
from gym.utils import seeding
import numpy as np

class TrajectoryGenerator(object):
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        pass

    def set_policy(self, policy):
        pass

    def set_horizon(self, horizon):
        self.horizon = horizon


class OnlineTrajectoryGenerator(TrajectoryGenerator):
    def __init__(self, mdp, policy):
        self.mdp = mdp
        self.policy = policy
        self.mdp.reset()
        self.horizon = mdp.horizon

    def set_policy(self, policy):
        self.policy = policy

    def next(self):
        return collect_episodes(self.mdp, self.policy, horizon=self.horizon)


class OfflineTrajectoryGenerator(TrajectoryGenerator):
    def __init__(self, dataset):
        self.horizon = 0
        self.dataset = []
        stop = np.where(dataset[:, -1] == 1)[0] + 1
        start = np.concatenate([[0], stop[:-1] - 1])
        for i,j in zip(start, stop):
            self.dataset.append(dataset[i:j])
            self.horizon = max(self.horizon, j-i)

        self.n_trajectories = len(self.dataset)
        self.seed()

    def next(self):
        return self.dataset[self.np_random.choice(self.n_trajectories)][:self.horizon]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)