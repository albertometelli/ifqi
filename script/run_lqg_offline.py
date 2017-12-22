from ifqi.envs.lqg1d import LQG1D
from ifqi.algorithms.policy_gradient.policy import GaussianPolicyLinearMean
from ifqi.evaluation.evaluation import collect_episodes
import numpy as np
import matplotlib.pyplot as plt


mdp = LQG1D()
K_opt = mdp.computeOptimalK()

sb = 2
st = 1
mub = -0.2
mut = -0.2
N = 1000
H = mdp.horizon
Hmin = 6
level = .95

#Instantiate policies
policy = GaussianPolicyLinearMean(mub, sb**2)
target = GaussianPolicyLinearMean(mut, st**2)

#Collect trajectories
dataset = collect_episodes(mdp, policy, n_episodes=N)

rmax = 0
rmin = (- mdp.max_action**2 * mdp.R - mdp.max_pos**2 * mdp.Q) *\
       (1 - mdp.gamma**N) / (1 - mdp.gamma)

s,a,r = np.array(map(lambda i: (
    np.reshape(dataset[:,i], (N, H))),
                 range(3)))

R = np.array(map(lambda i: (
    np.dot(r[i, :], np.power(mdp.gamma, range(0, H))) - rmin) / (rmax - rmin),
             range(N)))

#Compute Minf
def Minf(mub, mut, sb, st, max_pos):
    return ((sb / st) * np.exp ( 0.5 * (mut - mub)**2 * max_pos ** 2 / (sb**2 - st**2)))

#Compute optimal horizon
def Hstar(Minf, gamma, N, delta):
    return (int(round((1 / np.log(Minf)) *
            (np.log( (gamma * Minf - 1) / (np.log(gamma * Minf))) +
             np.log( np.log(gamma) / (gamma - 1)) +
             0.5 * (np.log(2 * N) - np.log(np.log(1/delta)))))))

#Compute weight matrix
def W_H(N, H, target, policy):
    W = np.zeros((N, H))

    for i in range(N):
        W[i, :] = np.array(map(lambda j:
                               target.pdf(np.array([s[i, j]]), np.array([a[i, j]])) /
                               policy.pdf(np.array([s[i, j]]), np.array([a[i, j]])),
                               range(H)))
    return (W)

#Compute estimate of J
def J_hat_H(R, W):
    return(R * np.prod(W, axis=1))

#Compute bound
def bound(N, H, R, mub, mut, sb, st, mdp, delta, target, policy):
    w_h = W_H(N, H, target, policy)
    j_hat = J_hat_H(R, w_h)
    minf = Minf(mub, mut, sb, st, mdp.max_pos)
    boundH = j_hat.mean() - (mdp.gamma**H)/(1-mdp.gamma) - \
        (1 - (mdp.gamma*minf)**H)/(1 - mdp.gamma*minf) * \
        np.sqrt(np.log(1/delta) / (2*N))
    return (boundH)

minf = Minf(mub, mut, sb, st, mdp.max_pos)
hstar = Hstar(minf, mdp.gamma, N, level)

boundh = bound(N, H, R, mub, mut, sb, st, mdp, level, target, policy)
boundstar = bound(N, hstar, R, mub, mut, sb, st, mdp, level, target, policy)

print("boundh ", boundh)
print("boundstar ", boundstar)

real_bound = [bound(N, h, R, mub, mut, sb, st, mdp, level, target, policy) for h in range(1,10)]
plt.plot(range(1,10), real_bound)

