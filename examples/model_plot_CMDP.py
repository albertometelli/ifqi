
import numpy as np
import matplotlib.pyplot as plt
from spmi.utils.tabular import *
from spmi.utils.tabular_operations import *
from spmi.utils.tabular_factory import *

from spmi.envs.simpleCMDP import SimpleCMDP
from spmi.envs.pathologicalCMDP import PathologicalCMDP


def value_iteration():
    V = np.zeros(mdp.nS)
    V_next = np.zeros(mdp.nS)
    P_sas = mdp.P_sas
    delta = float("inf")

    while delta > threshold:
        delta = 0
        V_sa = np.dot(P_sas, mdp.R + mdp.gamma * V)
        for s in range(mdp.nS):
            V_next[s] = np.max(V_sa[s])
        max_diff = max(np.absolute(V_next - V))
        delta = max(delta, max_diff)
        V = np.copy(V_next)

    return V


path_name = "/Users/mirco/Desktop/Simulazioni/optSPMI_simpleCMDP"
# path_name = "/Users/mirco/Desktop/Simulazioni/optSPMI_pathologicalCMDP"

grid_step = 0.01
threshold = 0.00001

mdp = SimpleCMDP(p=0.5, w=0.5)
# mdp = PathologicalCMDP(p=0.1, w=0.5, M=100)
model0 = TabularModel(mdp.P0, mdp.nS, mdp.nA)
model1 = TabularModel(mdp.P1, mdp.nS, mdp.nA)
matrix0 = model0.get_matrix()
matrix1 = model1.get_matrix()

J = list()
W = list()

for i in np.arange(0, 1 + grid_step, grid_step):
    new_matrix = i * matrix0 + (1 - i) * matrix1
    model = model_from_matrix(new_matrix, mdp.P)
    mdp.set_model(model.get_rep())
    V = value_iteration()
    J.append(np.dot(mdp.mu, V))
    W.append(i)
    print('iteration {0}'.format(i))

coefficient = np.array(W)
performance = np.array(J)

plt.switch_backend('pdf')
plt.figure()
plt.title('Optimal Model Performance')
plt.xlabel('Coefficient')
plt.ylabel('Performance')
plt.plot(coefficient, performance, label='model')
plt.legend(loc='best', fancybox=True)
plt.savefig(path_name + "/model_plot")
