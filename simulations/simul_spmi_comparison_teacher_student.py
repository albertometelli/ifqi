from spmi.envs.teacher_student import TeacherStudentEnv
from spmi.evaluation.evaluation import collect_episodes
from spmi.algorithms.spmi_exact_non_par import SPMI
from spmi.utils.uniform_policy import UniformPolicy
from spmi.utils.tabular import TabularPolicy, TabularModel
from spmi.algorithms.policy_chooser import *
from spmi.algorithms.model_chooser import *
import numpy as np
import copy
import matplotlib.pyplot as plt
import time
import cPickle
import os

path_name = 'data'

if __name__ == '__main__':

    startTime = time.time()

    mdp = TeacherStudentEnv(n_literals=3,
                        max_value=3,
                        max_update=3,
                        max_literals_in_examples=2,
                        horizon=10)

    simulation_name = '%s-%s-%s-%s-%s' % (mdp.n_literals,
                                  mdp.max_value,
                                  mdp.max_update,
                                  mdp.max_literals_in_examples,
                                  mdp.horizon)

    if not os.path.exists('data/' + simulation_name):
        os.makedirs('data/' + simulation_name)

    with open('data/' + simulation_name + '/' + simulation_name, 'wb') as pickle_file:
        cPickle.dump(mdp, pickle_file)

    uniform_policy = UniformPolicy(mdp)
    original_model = copy.deepcopy(mdp.P)

    initial_model = TabularModel(mdp.P, mdp.nS, mdp.nA)
    initial_policy = TabularPolicy(uniform_policy.get_rep(), mdp.nS, mdp.nA)

    policy_chooser = GreedyPolicyChooser(mdp.nS, mdp.nA)
    model_chooser = DoNotCreateTransitionsGreedyModelChooser(mdp.P, mdp.nS, mdp.nA)

    eps = 0.0
    spmi = SPMI(mdp, eps, policy_chooser, model_chooser, max_iter=30000, use_target_trick=True)

    #-------------------------------------------------------------------------------
    #SPMI
    policy, model = spmi.spmi(initial_policy, initial_model)

    iterations = np.array(range(spmi.iteration))
    evaluations = np.array(spmi.evaluations)
    p_advantages = np.array(spmi.p_advantages)
    m_advantages = np.array(spmi.m_advantages)
    p_dist_sup = np.array(spmi.p_dist_sup)
    p_dist_mean = np.array(spmi.p_dist_mean)
    m_dist_sup = np.array(spmi.m_dist_sup)
    m_dist_mean = np.array(spmi.m_dist_mean)
    alfas = np.array(spmi.alfas)
    betas = np.array(spmi.betas)
    p_change = np.cumsum(1 - np.array(spmi.p_change))
    m_change = np.cumsum(1 - np.array(spmi.m_change))
    spmi.save_simulation('data/'+ simulation_name, 'spmi.csv')

    #-------------------------------------------------------------------------------
    #SPMI sup
    mdp.set_model(original_model)
    policy, model = spmi.spmi_sup(initial_policy, initial_model)

    sup_iterations = np.array(range(spmi.iteration))
    sup_evaluations = np.array(spmi.evaluations)
    sup_p_advantages = np.array(spmi.p_advantages)
    sup_m_advantages = np.array(spmi.m_advantages)
    sup_p_dist_sup = np.array(spmi.p_dist_sup)
    sup_p_dist_mean = np.array(spmi.p_dist_mean)
    sup_m_dist_sup = np.array(spmi.m_dist_sup)
    sup_m_dist_mean = np.array(spmi.m_dist_mean)
    sup_alfas = np.array(spmi.alfas)
    sup_betas = np.array(spmi.betas)
    sup_p_change = np.cumsum(1 - np.array(spmi.p_change))
    sup_m_change = np.cumsum(1 - np.array(spmi.m_change))
    spmi.save_simulation('data/'+ simulation_name, 'sup.csv')

    #-------------------------------------------------------------------------------
    #SPMI no full step
    mdp.set_model(original_model)
    policy, model = spmi.spmi_no_full(initial_policy, initial_model)

    int_iterations = np.array(range(spmi.iteration))
    int_evaluations = np.array(spmi.evaluations)
    int_p_advantages = np.array(spmi.p_advantages)
    int_m_advantages = np.array(spmi.m_advantages)
    int_p_dist_sup = np.array(spmi.p_dist_sup)
    int_p_dist_mean = np.array(spmi.p_dist_mean)
    int_m_dist_sup = np.array(spmi.m_dist_sup)
    int_m_dist_mean = np.array(spmi.m_dist_mean)
    int_alfas = np.array(spmi.alfas)
    int_betas = np.array(spmi.betas)
    int_p_change = np.cumsum(1 - np.array(spmi.p_change))
    int_m_change = np.cumsum(1 - np.array(spmi.m_change))
    spmi.save_simulation('data/'+ simulation_name,  'nofull.csv')

    #-------------------------------------------------------------------------------
    #SPMI alternated
    mdp.set_model(original_model)
    policy, model = spmi.spmi_alt(initial_policy, initial_model)

    spmi.spmi_alt(initial_policy, initial_model)

    alt_iterations = np.array(range(spmi.iteration))
    alt_evaluations = np.array(spmi.evaluations)
    alt_p_advantages = np.array(spmi.p_advantages)
    alt_m_advantages = np.array(spmi.m_advantages)
    alt_p_dist_sup = np.array(spmi.p_dist_sup)
    alt_p_dist_mean = np.array(spmi.p_dist_mean)
    alt_m_dist_sup = np.array(spmi.m_dist_sup)
    alt_m_dist_mean = np.array(spmi.m_dist_mean)
    alt_alfas = np.array(spmi.alfas)
    alt_betas = np.array(spmi.betas)
    alt_p_change = np.cumsum(1 - np.array(spmi.p_change))
    alt_m_change = np.cumsum(1 - np.array(spmi.m_change))
    spmi.save_simulation('data/'+ simulation_name, 'alt.csv')

    #-------------------------------------------------------------------------------
    #SPMI sequential
    mdp.set_model(original_model)
    policy, model = spmi.spmi_seq_pm(initial_policy, initial_model)

    spmi.spmi_seq_pm(initial_policy, initial_model)

    seq_iterations = np.array(range(spmi.iteration))
    seq_evaluations = np.array(spmi.evaluations)
    seq_p_advantages = spmi.p_advantages
    seq_m_advantages = spmi.m_advantages
    seq_p_dist_sup = spmi.p_dist_sup
    seq_p_dist_mean = spmi.p_dist_mean
    seq_m_dist_sup = spmi.m_dist_sup
    seq_m_dist_mean = spmi.m_dist_mean
    seq_alfas = spmi.alfas
    seq_betas = spmi.betas
    seq_p_change = np.cumsum(1 - np.array(spmi.p_change))
    seq_m_change = np.cumsum(1 - np.array(spmi.m_change))
    spmi.save_simulation('data/'+ simulation_name, 'seq.csv')

    #-------------------------------------------------------------------------------
    #plots

    plt.switch_backend('pdf')

    plt.figure()
    plt.title('Performance')
    plt.xlabel('Iteration')
    plt.plot(iterations, evaluations, color='b', label='SPMI')
    plt.plot(sup_iterations, sup_evaluations, color='y', label='SPMI sup')
    plt.plot(alt_iterations, alt_evaluations, color='g', label='alternated')
    plt.plot(seq_iterations, seq_evaluations, color='r', label='sequential')
    plt.plot(int_iterations, int_evaluations, color='k', linestyle='dotted', label='no full step')
    plt.legend(loc='best', fancybox=True)
    plt.savefig(path_name + "/performance")

    plt.figure()
    plt.title('Policy distances')
    plt.xlabel('Iteration')
    plt.plot(iterations, p_dist_sup, color='b', label='SPMI sup')
    plt.plot(iterations, p_dist_mean, color='b', linestyle='dashed', label='SPMI mean')
    plt.plot(sup_iterations, sup_p_dist_sup, color='y', label='sup sup')
    plt.plot(sup_iterations, sup_p_dist_mean, color='y', linestyle='dashed', label='sup mean')
    plt.plot(alt_iterations, alt_p_dist_sup, color='g', label='alternated sup')
    plt.plot(alt_iterations, alt_p_dist_mean, color='g', linestyle='dashed', label='alternated mean')
    plt.plot(seq_iterations, seq_p_dist_sup, color='r', label='sequential sup')
    plt.plot(seq_iterations, seq_p_dist_mean, color='r', linestyle='dashed', label='sequential mean')
    plt.plot(int_iterations, int_p_dist_sup, color='k', linestyle='dotted', label='no full step sup')
    plt.plot(int_iterations, int_p_dist_mean, color='k', linestyle='dashed', label='no full step mean')
    plt.legend(loc='best', fancybox=True)
    plt.savefig(path_name  + "/policy distance")

    plt.figure()
    plt.title('Model distances')
    plt.xlabel('Iteration')
    plt.plot(iterations, m_dist_sup, color='b', label='SPMI sup')
    plt.plot(iterations, m_dist_mean, color='b', linestyle='dashed', label='SPMI mean')
    plt.plot(sup_iterations, sup_m_dist_sup, color='y', label='sup sup')
    plt.plot(sup_iterations, sup_m_dist_mean, color='y', linestyle='dashed', label='sup mean')
    plt.plot(alt_iterations, alt_m_dist_sup, color='g', label='alternated sup')
    plt.plot(alt_iterations, alt_m_dist_mean, color='g', linestyle='dashed', label='alternated mean')
    plt.plot(seq_iterations, seq_m_dist_sup, color='r', label='sequential sup')
    plt.plot(seq_iterations, seq_m_dist_mean, color='r', linestyle='dashed', label='sequential mean')
    plt.plot(int_iterations, int_m_dist_sup, color='k', linestyle='dotted', label='np full step sup')
    plt.plot(int_iterations, int_m_dist_mean, color='k', linestyle='dashed', label='no full step mean')
    plt.legend(loc='best', fancybox=True)
    plt.savefig(path_name + "/model distance")

    plt.figure()
    plt.title('Advantage')
    plt.xlabel('Iteration')
    plt.plot(iterations, p_advantages, color='b', label='SPMI policy')
    plt.plot(iterations, m_advantages, color='b', linestyle='dashed', label='SPMI model')
    plt.plot(sup_iterations, sup_p_advantages, color='y', label='sup policy')
    plt.plot(sup_iterations, sup_m_advantages, color='y', linestyle='dashed', label='sup model')
    plt.plot(alt_iterations, alt_p_advantages, color='g', label='alternated policy')
    plt.plot(alt_iterations, alt_m_advantages, color='g', linestyle='dashed', label='alternated model')
    plt.plot(seq_iterations, seq_p_advantages, color='r', label='sequential policy')
    plt.plot(seq_iterations, seq_m_advantages, color='r', linestyle='dashed', label='sequential model')
    plt.plot(int_iterations, int_p_advantages, color='k', linestyle='dotted', label='no full step policy')
    plt.plot(int_iterations, int_m_advantages, color='k', linestyle='dashed', label='no full step model')
    plt.legend(loc='best', fancybox=True)
    plt.savefig(path_name + "/advantages")

    plt.figure()
    plt.title('Alfa & Beta')
    plt.xlabel('Iteration')
    plt.plot(iterations, alfas, color='b', linestyle='', marker='o', label='SPMI alfa')
    plt.plot(iterations, betas, color='b', linestyle='', marker='s', label='SPMI beta')
    plt.plot(sup_iterations, sup_alfas, color='y', linestyle='', marker='o', label='sup alfa')
    plt.plot(sup_iterations, sup_betas, color='y', linestyle='', marker='s', label='sup beta')
    plt.plot(alt_iterations, alt_alfas, color='g', linestyle='', marker='o', label='alternated alfa')
    plt.plot(alt_iterations, alt_betas, color='g', linestyle='', marker='s', label='alternated beta')
    plt.plot(seq_iterations, seq_alfas, color='r', linestyle='', marker='o', label='sequential alfa')
    plt.plot(seq_iterations, seq_betas, color='r', linestyle='', marker='s', label='sequential beta')
    plt.plot(int_iterations, int_alfas, color='k', linestyle='', marker='o', label='no full step alfa')
    plt.plot(int_iterations, int_betas, color='k', linestyle='', marker='s', label='no full step beta')
    plt.legend(loc='best', fancybox=True)
    plt.yscale('log')
    plt.savefig(path_name + "/alfabeta")

    plt.figure()
    plt.title('Policy Model Changes')
    plt.xlabel('Iteration')
    plt.plot(iterations, p_change, color='b', label='SPMI policy')
    plt.plot(iterations, m_change, color='b', linestyle='dashed', label='SPMI model')
    plt.plot(sup_iterations, sup_p_change, color='y', label='sup policy')
    plt.plot(sup_iterations, sup_m_change, color='y', linestyle='dashed', label='sup model')
    plt.plot(alt_iterations, alt_p_change, color='g', label='alternated policy')
    plt.plot(alt_iterations, alt_m_change, color='g', linestyle='dashed', label='alternated model')
    plt.plot(seq_iterations, seq_p_change, color='r', label='sequential policy')
    plt.plot(seq_iterations, seq_m_change, color='r', linestyle='dashed', label='sequential model')
    plt.plot(int_iterations, int_p_change, color='k', linestyle='dotted', label='no full step policy')
    plt.plot(int_iterations, int_m_change, color='k', linestyle='dashed', label='no full step model')
    plt.legend(loc='best', fancybox=True)
    plt.savefig(path_name + "/changes")



    print('The script took {0} minutes'.format((time.time() - startTime) / 60))