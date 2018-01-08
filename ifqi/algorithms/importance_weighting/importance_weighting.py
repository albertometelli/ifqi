import numpy as np

class ImportanceWeighting(object):

    def __init__(self,
                 behavioral_policy,
                 target_policy=None,
                 state_index=0,
                 action_index=1):

        self.behavioral_policy = behavioral_policy
        self.target_policy = target_policy
        self.state_index = state_index
        self.action_index = action_index

    def set_target_policy(self, target_policy):
        self.target_policy = target_policy

    def weight(self, trajectory):
        pass

class DummyImportanceWeighting(ImportanceWeighting):
    def weight(self, trajectory):
        return np.ones(len(trajectory))


class RatioImportanceWeighting(ImportanceWeighting):

    def weight(self, trajectory):
        w = 1.
        for i in range(len(trajectory)):

            w *= self.target_policy.pdf(trajectory[i, self.state_index], \
                        trajectory[i, self.action_index]) / \
                  self.behavioral_policy.pdf(trajectory[i, self.state_index], \
                        trajectory[i, self.action_index])

        return np.repeat(w, len(trajectory))

class PerDecisionRatioImportanceWeighting(ImportanceWeighting):

    def weight(self, trajectory):
        w = np.zeros(len(trajectory))
        w[0] = self.target_policy.pdf(trajectory[0, self.state_index], \
                                        trajectory[0, self.action_index]) / \
               self.behavioral_policy.pdf(trajectory[0, self.state_index], \
                                        trajectory[0, self.action_index])
        for i in range(1, len(trajectory)):
            w[i] = w[i-1] * self.target_policy.pdf(trajectory[i, self.state_index], \
                                        trajectory[i, self.action_index]) / \
                 self.behavioral_policy.pdf(trajectory[i, self.state_index], \
                                        trajectory[i, self.action_index])

        return w