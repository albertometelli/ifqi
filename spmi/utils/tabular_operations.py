import tabular_factory
import numpy as np

def policy_convex_combination(policy1, policy2, coeff):

    if coeff < 0 or coeff > 1:
        raise ValueError()

    policy1_matrix = policy1.get_matrix()
    policy2_matrix = policy2.get_matrix()

    matrix = coeff * policy1_matrix + (1. - coeff) * policy2_matrix

    return tabular_factory.policy_from_matrix(matrix)

def model_convex_combination(original_model, model1, model2, coeff):

    if coeff < 0 or coeff > 1:
        raise ValueError()

    model1_matrix = model1.get_matrix()
    model2_matrix = model2.get_matrix()

    matrix = coeff * model1_matrix + (1. - coeff) * model2_matrix

    return tabular_factory.model_from_matrix(matrix, original_model)

# method to compute the infinite norm between two given policies
def policy_sup_tv_distance(policy1, policy2):
    policy1_matrix = policy1.get_matrix()
    policy2_matrix = policy2.get_matrix()
    return np.max(np.sum(np.abs(policy1_matrix - policy2_matrix), axis=1))

# method to compute the mean value distance between two given policies
def policy_mean_tv_distance(policy1, policy2, d_mu_pi):
    policy1_matrix = policy1.get_matrix()
    policy2_matrix = policy2.get_matrix()
    return np.dot(d_mu_pi, np.sum(np.abs(policy1_matrix - policy2_matrix), axis=1))

# method to compute the infinite norm between two given models
def model_sup_tv_distance(P1_sa, P2_sa):
    P1_sa_matrix = P1_sa.get_matrix()
    P2_sa_matrix = P2_sa.get_matrix()
    return np.max(np.sum(np.abs(P1_sa_matrix - P2_sa_matrix), axis=1))

# method to compute the mean value distance between two given models
def model_mean_tv_distance(P1_sa, P2_sa, delta_mu):
    P1_sa_matrix = P1_sa.get_matrix()
    P2_sa_matrix = P2_sa.get_matrix()
    return np.dot(delta_mu, (np.sum(np.abs(P1_sa_matrix - P2_sa_matrix), axis=1)))