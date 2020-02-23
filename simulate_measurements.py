import sys
import os

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

from modules.fuzzy_control_system import map_variable_types, create_rule_control_system, apply_rules
from modules.fuzzy_load import *
from modules.fuzzy_membership import create_membership_functions


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


def sample_defuzz(file, antc_i='HR', antc_j='R', step_size=5):
    '''
    Inferences the results for each possible sample from the two anticedents with a fixed step size.
    
        Args:
            file(str): the knowledge base file name
			antc_i(str): the variable name of the first anticedent (ex. 'HR')
			antc_j(str): the variable name of the second anticedent (ex. 'R')
			step_size(int): number of steps to skip before displaying the generated fuzzy set (def: 100)
			
		Returns:
			fuzzy_result_list(list): the list of simulation results
			fuzzy_sample_list(list): the list of sampled membership functions
			n_samples(int): count of inferenced samples
            
    '''
    print('--- Simulating all defuzzified values in the target sets ---')
    fuzzy_sample_list = []
    fuzzy_result_list = []
    n_samples = 0
    fuzzy_measurement_dict = {};

    fuzzy_dict, x_ranges, var_names, fuzzy_variables = create_membership_functions(file)
    for i in tqdm(range(1, len(x_ranges[antc_i]), step_size)):
        for j in range(1, len(x_ranges[antc_j]), step_size):
            fuzzy_measurement_dict[antc_i] = np.float32(x_ranges[antc_i][i])
            fuzzy_measurement_dict[antc_j] = np.float32(x_ranges[antc_j][j])
            vmfx_list, _ = map_variable_types(file, fuzzy_variables, var_names, x_ranges, fuzzy_dict)
            rcs = create_rule_control_system(file, fuzzy_variables, var_names, vmfx_list)
            try:
                ctr_sys_sim, consequent = apply_rules(rcs, fuzzy_measurement_dict, var_names, vmfx_list)
            except ValueError:
                continue
            fuzzy_sample_list.append(fuzzy_measurement_dict)
            fuzzy_measurement_dict = {}
            fuzzy_result_list.append(ctr_sys_sim.output)
            n_samples += 1

    print('Sample size:', n_samples)
    return fuzzy_sample_list, fuzzy_result_list, n_samples


def plot_simulated_measurements(fuzzy_sample_list, fuzzy_result_list, n_samples, antc_i='HR', antc_j='R'):
    '''
    Plots the measurement values and defuzzification results with respect to the number of iterations.
    
        Args:
            fuzzy_result_list(list): the list of simulation results
			fuzzy_sample_list(list): the list of sampled membership functions
			n_samples(int): count of inferenced samples
			antc_i(str): the variable name of the first anticedent (ex. 'HR')
			antc_j(str): the variable name of the second anticedent (ex. 'R')

    '''
    ms_i = [];
    ms_j = [];
    for fuzzy_sample in fuzzy_sample_list:
        for k, v in fuzzy_sample.items():
            if k == antc_i:
                ms_i.append(v)
            elif k == antc_j:
                ms_j.append(v)

    fz_lbl = ""
    dfz_values = []
    for fz in fuzzy_result_list:
        for k, v in fz.items():
            if fz_lbl == "":
                fz_lbl = k
            dfz_values.append(v)

    fig, axs = plt.subplots(3, figsize=(8, 6))
    sns.lineplot(ms_i, np.arange(1, n_samples + 1), color='red', ax=axs[0])
    axs[0].set_ylabel('Iterations', fontsize=14)
    axs[0].set_xlabel(antc_i, fontsize=14)
    axs[0].lines[0].set_linestyle("--")
    sns.lineplot(ms_j, np.arange(1, n_samples + 1), color='orange', ax=axs[1])
    axs[1].set_ylabel('Iterations', fontsize=14)
    axs[1].set_xlabel(antc_j, fontsize=14)
    axs[1].lines[0].set_linestyle("--")
    sns.lineplot(dfz_values, np.arange(1, n_samples + 1), color='darkgreen', ax=axs[2])
    axs[2].set_ylabel('Iterations', fontsize=14)
    axs[2].set_xlabel(str('Defuzzified value: ' + fz_lbl), fontsize=14)
    plt.subplots_adjust(hspace=0.66)
    plt.show()

    print('Anticedent {} statistics:{}'.format(antc_i, stats.describe(ms_i)))
    print('Anticedent {} statistics:{}'.format(antc_j, stats.describe(ms_j)))
    print('Consequent {} statistics:{}'.format(fz_lbl, stats.describe(dfz_values)))


if __name__ == '__main__':
    with HiddenPrints():
        # argv[1] = input file name, argv[2] = variable name of anticedent 1, argv[3] = variable name of anticedent 2, argv[4] = step size (def. 10)
        fuzzy_sample_list, fuzzy_result_list, n_samples = sample_defuzz(sys.argv[1], sys.argv[2], sys.argv[3],
                                                                        int(sys.argv[4]))
    plot_simulated_measurements(fuzzy_sample_list, fuzzy_result_list, n_samples, sys.argv[2], sys.argv[3])
