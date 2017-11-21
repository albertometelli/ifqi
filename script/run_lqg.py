from ifqi.envs.lqg1d import LQG1D
from ifqi.algorithms.policy_gradient.policy import GaussianPolicy
from ifqi.evaluation.evaluation import collect_episodes
import numpy as np
import scipy.stats


mdp = LQG1D()
K_opt = mdp.computeOptimalK()

sb = 13
st = 6
mub = -0.3
mut = -.1
N = 100
H = 6

policy = GaussianPolicy (mub, sb**2)

dataset = collect_episodes (mdp, policy, n_episodes=N)

rmax = 0
rmin = (- mdp.max_action**2 * mdp.R - mdp.max_pos**2 * mdp.Q) *\
       (1 - mdp.gamma**N) / (1 - mdp.gamma)

s = np.reshape (dataset[:,0], (N, H))
a = np.reshape (dataset[:,1], (N, H))
r = np.reshape (dataset[:,2], (N, H))
R = np.zeros (N)

for i in range (N):
    R[i] = (np.dot (r[i, :], np.power (mdp.gamma, range (0, H))) - rmin) / (rmax - rmin)

W = np.zeros ((N, H))

target = GaussianPolicy (mut, st**2)

for i in range (N):

    for j in range (H):

        W[i,j] = target.pdf (np.array ([s[i, j]]), np.array ([a[i, j]])) / \
                 policy.pdf (np.array ([s[i, j]]), np.array ([a[i, j]]))

level = .95

alpha = np.sqrt (scipy.stats.norm.ppf (level)**2 / N)
J_hat = np.zeros (N)
w = 1


renyi = (sb**2 / (st * np.sqrt (2 * sb**2 - st**2))) *\
        np.exp ((mut - mub)**2 * 100 / (2 * sb**2 - st**2))

J_hat = R * np.prod(W, axis=1)

bound_renyi = (J_hat.mean () -
         alpha * np.sqrt ( - J_hat.mean()**2 + (1 + alpha**2) * renyi**H)) /\
        (alpha**2 + 1)

wmax = (sb / st) * np.exp ( 0.5 * (mut - mub)**2 * 100 / (sb**2 - st**2))

bound_max = (alpha * wmax**H * 0.5 + J_hat.mean() -
              np.sqrt (alpha**2 * wmax**(2*H) + 4 * alpha * wmax**H * J_hat.mean() -
                      4 * alpha**2 * J_hat.mean()**2)) /\
            (alpha**2 + 1)

bound_t = J_hat.mean() - np.sqrt(J_hat.var()) * scipy.stats.t.ppf(level, N-1)

print("bound_renyi ", bound_renyi, "bound_max ", bound_max, "bound_t ", bound_t)


wmax_trunc = (sb / st) * np.exp ( 0.5 * (mut - mub)**2 * 100 / (sb**2 - st**2)) * \
             np.exp( -0.5 * ((sb**2 - st**2) / (sb**2 * st**2)) *
                  (np.min((mdp.max_pos, (sb**2 * mut - st**2 * mub) / (sb**2 - st**2))) -
                   (sb**2 * mut - st**2 * mub) / (sb**2 - st**2)))

bound_max_trunc = (alpha * wmax_trunc**H * 0.5 + J_hat.mean() -
              np.sqrt (alpha**2 * wmax_trunc**(2*H) + 4 * alpha * wmax_trunc**H * J_hat.mean() -
                      4 * alpha**2 * J_hat.mean()**2)) /\
            (alpha**2 + 1)
## Compute renyi divergence for truncated gaussian


phi1 = scipy.stats.norm.cdf(mdp.max_pos, mut, st)

phi2 = scipy.stats.norm.cdf(mdp.max_pos, mub, sb)

phi3 = scipy.stats.norm.cdf(mdp.max_pos,
                            (2 * sb**2 * mut - st**2 * mub) / (2 * sb**2 - st**2),
                            (sb ** 2 * st ** 2) / (2 * sb ** 2 - st ** 2))

scale = (1 + phi1) * (1 + phi3) / (1 + phi2)**2

bound_renyi_trunc = (J_hat.mean () -
         alpha * np.sqrt ( - J_hat.mean()**2 + (1 + alpha**2) * (renyi * scale)**H)) /\
        (alpha**2 + 1)

print("bound_renyi_trunc ", bound_renyi_trunc,
      "bound_max_trunc ", bound_max_trunc)
