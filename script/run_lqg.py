from ifqi.envs.lqg1d import LQG1D
from ifqi.algorithms.policy_gradient.policy import GaussianPolicy
from ifqi.evaluation.evaluation import collect_episodes
import numpy as np
import scipy.stats


mdp = LQG1D()
K_opt = mdp.computeOptimalK()

sb = 2
st = 1
mub = -0.3
mut = -.1
N = 1000
H = mdp.horizon

#Instantiate policies
policy = GaussianPolicy(mub, sb**2)
target = GaussianPolicy(mut, st**2)

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

#Compute weight matrix
W= np.zeros((N, H))

for i in range (N):
    W[i,:] = np.array(map(lambda j:
        target.pdf(np.array([s[i, j]]), np.array([a[i, j]])) / \
        policy.pdf(np.array([s[i, j]]), np.array([a[i, j]])),
                      range(H)))

level = .95

def bound(R, W, sb, mub, st, mut, N, H, mdp):
    J_hat = R * np.prod(W, axis=1)
    MinfH = (sb / st) * np.exp ( 0.5 * (mut - mub)**2 * mdp.max_pos ** 2 / (sb**2 - st**2))
    boundH = J_hat.mean() - (mdp.gamma**H)/(1-mdp.gamma) - \
        (1 - (mdp.gamma*MinfH)**H)/(1 - mdp.gamma*MinfH) * \
        np.sqrt(np.log(1/level) / (2*N))
    return (boundH)


print("bound ", bound(R, W, sb, mub, st, mut, N, H, mdp))

