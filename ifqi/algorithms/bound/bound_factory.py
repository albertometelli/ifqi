from ifqi.algorithms.bound.bound import *
from ifqi.algorithms.importance_weighting.importance_weighting import *

def build_Hoeffding_bound(is_method, N, delta, gamma, behavioral_policy, target_policy):
    if isinstance(is_method, PerDecisionRatioImportanceWeighting):
        return HoeffdingBoundPerDecisionRatioImportanceWeighting(N, delta, gamma,
                                                      behavioral_policy,
                                                      target_policy)
    if isinstance(is_method, RatioImportanceWeighting):
        return HoeffdingBoundRatioImportanceWeighting(N, delta, gamma,
                                                      behavioral_policy,
                                                      target_policy)

    raise NotImplementedError()

def build_Chebyshev_bound(is_method, N, delta, gamma, behavioral_policy, target_policy):
    if isinstance(is_method, RatioImportanceWeighting):
        return ChebyshevBoundRatioImportanceWeighting(N, delta, gamma,
                                                      behavioral_policy,
                                                      target_policy)
    raise NotImplementedError()

def build_Bernstein_bound(is_method, N, delta, gamma, behavioral_policy, target_policy):
    if isinstance(is_method, RatioImportanceWeighting):
        return BernsteinBoundRatioImportanceWeighting(N, delta, gamma,
                                                      behavioral_policy,
                                                      target_policy)
    raise NotImplementedError()