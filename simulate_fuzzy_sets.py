import random
import os
import sys

import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

from modules.fuzzy_control_system import map_variable_types, create_rule_control_system, apply_rules, view_defuzz
from modules.fuzzy_load import *
from modules.fuzzy_membership import create_membership_functions

sns.set(style='darkgrid', palette="Paired")


class HiddenPrints:
    '''
    Hides printed values for debugging.
    '''
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')


    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def sample_fuzzy(file, keep_prob=0.5, n_iter=1000, step_size=100, conseq_var='D'):
    '''
    Randomly samples different variations of fuzzy sets for each variable category with a certain probability.
    
        Args:
            file(str): the knowledge base file name
            keep_prob(float): probability to resample the original set for each iteration (def: 0.5)
            n_iter(int): number of iterations of the simulation (def: 1000)
            step_size(int): number of steps to skip before displaying the generated fuzzy set (def: 100) conseq_var(str): the name of the consequent variable (ex. 'D')

        Returns:
            fuzzy_result_list(list): the list of simulation results
            valid_samples(int): count of valid samples generated
            fuzzy_sample_list(list): the list of sampled membership functions
            
    '''


    print('--- Random sampling fuzzy sets ---')
    fuzzy_result_list = []
    valid_samples = 0
    fuzzy_sample_list = []

    for i in tqdm(range(n_iter)):
        fuzzy_variables = read_variables(file)
        for k, v in fuzzy_variables.items():
            if k == conseq_var:
                continue
            for k_j, v_j in v.items():
                rnd = random.random()
                max_range = max(v_j)
                if rnd > keep_prob:
                    gen_a = np.random.randint(0, max_range + 1, 1).item(0)
                    gen_b = np.random.randint(0, max_range + 1, 1).item(0)
                    gen_alpha = np.random.randint(0, max_range + 1, 1).item(0)
                    gen_beta = np.random.randint(0, max_range + 1, 1).item(0)
                    if gen_b >= fuzzy_variables[k][k_j][0]:
                        fuzzy_variables[k][k_j] = list(fuzzy_variables[k][k_j])
                        fuzzy_variables[k][k_j][1] = gen_b
                        fuzzy_variables[k][k_j] = tuple(fuzzy_variables[k][k_j])
                    if gen_alpha <= fuzzy_variables[k][k_j][0]:
                        fuzzy_variables[k][k_j] = list(fuzzy_variables[k][k_j])
                        fuzzy_variables[k][k_j][2] = gen_alpha
                        fuzzy_variables[k][k_j] = tuple(fuzzy_variables[k][k_j])
                    if gen_beta <= fuzzy_variables[k][k_j][1]:
                        fuzzy_variables[k][k_j] = list(fuzzy_variables[k][k_j])
                        fuzzy_variables[k][k_j][3] = gen_beta
                        fuzzy_variables[k][k_j] = tuple(fuzzy_variables[k][k_j])
                    if gen_a <= fuzzy_variables[k][k_j][1]:
                        fuzzy_variables[k][k_j] = list(fuzzy_variables[k][k_j])
                        fuzzy_variables[k][k_j][0] = gen_a
                        fuzzy_variables[k][k_j] = tuple(fuzzy_variables[k][k_j])

        fuzzy_dict, x_ranges, var_names, _ = create_membership_functions(fuzzy_variables, from_file=False)
        vmfx_list, fuzzy_measurements = map_variable_types(file, fuzzy_variables, var_names, x_ranges,
                                                           fuzzy_dict)
        rcs = create_rule_control_system(file, fuzzy_variables, var_names, vmfx_list)
        try:
            ctr_sys_sim, consequent = apply_rules(rcs, fuzzy_measurements, var_names, vmfx_list)
            if (valid_samples + 1) % step_size == 0:
                view_defuzz(consequent, ctr_sys_sim)
        except ValueError:
            continue

        fuzzy_result_list.append(ctr_sys_sim.output)
        valid_samples += 1
        fuzzy_sample_list.append(fuzzy_variables)

    return fuzzy_result_list, valid_samples, fuzzy_sample_list


def plot_simulated_defuzz(fuzzy_result_list, samples):
    '''
    Plots the defuzzified value for each valid iteration.
    
        Args:
            fuzzy_result_list(list): the list of simulation results
            samples(int): count of valid samples generated
            
    '''


    fz_lbl = ""
    dfz_values = []
    for fz in fuzzy_result_list:
        for k, v in fz.items():
            if fz_lbl == "":
                fz_lbl = k
            dfz_values.append(v)

    plt.figure(figsize=(8, 6))
    plt.title('Defuzzified values')
    sns.lineplot(np.arange(1, samples + 1), dfz_values, color='darkgreen')
    plt.xlabel('Valid Sample size', fontsize=14)
    plt.ylabel(str('Defuzzified value: ' + fz_lbl), fontsize=14)

    plt.show()
    print('Consequent {} statistics:{}'.format(fz_lbl, stats.describe(dfz_values)))

if __name__ == '__main__':
    with HiddenPrints():
        # argv[1] = input file name, argv[2] = variable name of the target consequent, argv[3] = step size (def. 200)
        fuzzy_result_list, samples, fuzzy_sample_list = sample_fuzzy(sys.argv[1], conseq_var=sys.argv[2],
                                                                     step_size=int(sys.argv[3]))
    plot_simulated_defuzz(fuzzy_result_list, samples)
