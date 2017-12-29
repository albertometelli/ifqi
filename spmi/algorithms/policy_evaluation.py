import numpy as np

# method to exactly evaluate a policy in terms of q-function
# it returns Q: dictionary over states indexing array of values
def evaluate_policy(policy, model, reward, gamma, threshold):

    # initializations
    nS = mdp.nS
    nA = mdp.nA
    P = mdp.P
    gamma = mdp.gamma
    policy_rep = policy.get_rep()

    # tabular q-function instantiation
    Q = {s: np.zeros(nA) for s in range(nS)}
    Qnext = {s: np.zeros(nA) for s in range(nS)}

    # evaluation loop
    delta = 1
    while delta > threshold:
        delta = 0
        for s in range(nS):
            valid = mdp.get_valid_actions(s)
            for a in valid:
                sum = 0
                temp = Q[s][a]
                ns_list = P[s][a]
                for elem in ns_list:
                    p = elem[0]
                    ns = elem[1]
                    r = elem[2]
                    pi_arr = policy_rep[ns]
                    q_arr = Q[ns]
                    next_ret = np.dot(pi_arr, q_arr)
                    sum = sum + p * (r + gamma * next_ret)
                Qnext[s][a] = sum
                delta = max(delta, np.abs(temp - sum))
        Q = Qnext

    return Q

def evaluate_policy(mdp, policy, threshold):

    # initializations
    nS = mdp.nS
    nA = mdp.nA
    P = mdp.P
    gamma = mdp.gamma
    policy_rep = policy.get_rep()

    # tabular q-function instantiation
    Q = {s: np.zeros(nA) for s in range(nS)}
    Qnext = {s: np.zeros(nA) for s in range(nS)}

    # evaluation loop
    delta = 1
    while delta > threshold:
        delta = 0
        for s in range(nS):
            valid = mdp.get_valid_actions(s)
            for a in valid:
                sum = 0
                temp = Q[s][a]
                ns_list = P[s][a]
                for elem in ns_list:
                    p = elem[0]
                    ns = elem[1]
                    r = elem[2]
                    pi_arr = policy_rep[ns]
                    q_arr = Q[ns]
                    next_ret = np.dot(pi_arr, q_arr)
                    sum = sum + p * (r + gamma * next_ret)
                Qnext[s][a] = sum
                delta = max(delta, np.abs(temp - sum))
        Q = Qnext

    return Q