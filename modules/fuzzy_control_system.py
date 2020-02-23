### Scikit-fuzzy

import seaborn as sns
import matplotlib.pyplot as plt

from modules.fuzzy_load import *

sns.set(style='darkgrid', palette="Paired")
from skfuzzy import control as ctrl


def map_variable_types(measurement_file, fuzzy_variables, var_names, x_ranges, fuzzy_dict):
    '''
    Creates a lookup mapping of the current fuzzy variables using skfuzzy ctrl objects (Anticedent and Consequent).
    
        Args:
            measurement_file(str): the input knowledge base file name
            fuzzy_variables(dict): the processed fuzzy dictionary with memberships
            var_names(list): lookup list with variable names
            x_ranges(dict): membership ranges for each fuzzy variable
            fuzzy_dict(dict): the original parsed fuzzy variable dictionary

        Returns:
            var_type_list(list): list of dictionaries containing the variable name, type and range
            fuzzy_measurements(dict): parsed dictionary of measurements
            
    '''


    var_type_list = []
    fuzzy_measurements = read_measurements(measurement_file, fuzzy_variables)
    for var_name in var_names:
        if var_name in fuzzy_measurements.keys():
            antecedent = ctrl.Antecedent(x_ranges[var_name], var_name)
            var_type_list.append(antecedent)
        else:
            consequent = ctrl.Consequent(x_ranges[var_name], var_name)
            var_type_list.append(consequent)

    for k, v in fuzzy_dict.items():
        if k in var_names:
            for k_j, v_j in v.items():
                for vmfx in var_type_list:
                    if vmfx.label == k:
                        vmfx[str(k_j)] = v_j

    return var_type_list, fuzzy_measurements


def view_sample_set(fuzzy_set, target_category):
    '''
    Plots a sample fuzzy variable with an underlined target set.

        Returns:
            fuzzy_set[target_category].view()(matplotlib object): the target plot
    '''


    return fuzzy_set[target_category].view()


def create_rule_control_system(file, fuzzy_vars, var_names, vmfx_list):
    '''
    Creates skfuzzy Rule objects based of the dictionary of rules.
    
        Args:
            file(str): the input knowledge base file name
            fuzzy_vars(dict): the processed fuzzy dictionary with memberships
            var_names(list): lookup list with variable names
            vmfx_list(list): list of dictionaries containing the variable names, ranges and membership functions
            
        Returns:
            rcs(list): a list of dictionaries representing the rules within the control system
            
    '''
    # can handle up to 5 consecutive AND or OR connectors


    fuzzy_rules = read_rulebase(file, fuzzy_vars)
    rcs = []
    idx = 1
    for fuzzy_dict in fuzzy_rules:
        if fuzzy_dict['connector'] == 'AND' or fuzzy_dict['connector'] == 'OR':
            rule_prec_list = []
            result = ""
            for k_j, v_j in fuzzy_dict['precedents'].items():
                if k_j in var_names:
                    for vmfx in vmfx_list:
                        if vmfx.label == k_j:
                            rule_prec_list.append(vmfx[str(v_j)])

            for k_r, v_r in fuzzy_dict['result'].items():
                if k_r in var_names:
                    for vmfx in vmfx_list:
                        if vmfx.label == k_r:
                            result = vmfx[str(v_r)]

                if fuzzy_dict['connector'] == 'AND':
                    if len(rule_prec_list) == 2:
                        rule_i = ctrl.Rule(rule_prec_list[0] & rule_prec_list[1], result, label='R' + str(idx))
                    elif len(rule_prec_list) == 3:
                        rule_i = ctrl.Rule(rule_prec_list[0] & rule_prec_list[1] & rule_prec_list[2], result,
                                           label='R' + str(idx))
                    elif len(rule_prec_list) == 4:
                        rule_i = ctrl.Rule(
                            rule_prec_list[0] & rule_prec_list[1] & rule_prec_list[2] & rule_prec_list[3], result,
                            label='R' + str(idx))
                    elif len(rule_prec_list) == 5:
                        rule_i = ctrl.Rule(
                            rule_prec_list[0] & rule_prec_list[1] & rule_prec_list[2] & rule_prec_list[3] &
                            rule_prec_list[4], result, label='R' + str(idx))
                elif fuzzy_dict['connector'] == 'OR':
                    if len(rule_prec_list) == 2:
                        rule_i = ctrl.Rule(rule_prec_list[0] | rule_prec_list[1], result, label='R' + str(idx))
                    elif len(rule_prec_list) == 3:
                        rule_i = ctrl.Rule(rule_prec_list[0] | rule_prec_list[1] | rule_prec_list[2], result,
                                           label='R' + str(idx))
                    elif len(rule_prec_list) == 4:
                        rule_i = ctrl.Rule(
                            rule_prec_list[0] | rule_prec_list[1] | rule_prec_list[2] | rule_prec_list[3], result,
                            label='R' + str(idx))
                    elif len(rule_prec_list) == 5:
                        rule_i = ctrl.Rule(
                            rule_prec_list[0] | rule_prec_list[1] | rule_prec_list[2] | rule_prec_list[3] |
                            rule_prec_list[4], result, label='R' + str(idx))
                else:
                    print('Input rule is too complex to be translated')
                    return
                idx += 1

            if rule_i is not None:
                rcs.append(rule_i)
        elif fuzzy_dict['connector'] == 'SIMPLE':
            rule_prec = ""
            result = ""
            for k_j, v_j in fuzzy_dict['precedents'].items():
                if k_j in var_names:
                    for vmfx in vmfx_list:
                        if vmfx.label == k_j:
                            rule_prec = vmfx[str(v_j)]
            for k_r, v_r in fuzzy_dict['result'].items():
                if k_r in var_names:
                    for vmfx in vmfx_list:
                        if vmfx.label == k_r:
                            result = vmfx[str(v_r)]

            rule_i = ctrl.Rule(rule_prec, result, label='R' + str(idx))
            rcs.append(rule_i)
            idx += 1

    return rcs


def plot_rule_graphs(rule_list):
    '''
    Creates a DAG visualization of the rule relationships using skfuzzy's integrated networkx module.
    
        Args:
            rule_list(list): a list of dictionaries representing the rules within the control system
            
    '''


    for rule in rule_list:
        rule.view()
        plt.show()


def apply_rules(rcs, fuzzy_measurements, var_names, vmfx_list):
    '''
    Creates a ControlSystemSimulation object based on the list of rule objects.
    Performs automatic rule inference and defuzzification using the centroid method.

        Args:
            rcs(list): a list of dictionaries representing the rules within the control system
            fuzzy_measurements(dict): the dictionary of parsed measurements
            var_names(list): lookup list with variable names
            vmfx_list(list): list of dictionaries containing the variable names, ranges and membership functions

        Returns:
            ctr_sys_sim(ctrl.ControlSystemSimulation): the simulation object
            target_consequent(str): the target fuzzy variable name
            
    '''


    ctrl_sys = ctrl.ControlSystem(rcs)
    ctr_sys_sim = ctrl.ControlSystemSimulation(ctrl_sys)
    ctr_sys_sim.reset()
    target_consequent = ""
    print('--- Measurements ---')
    for k, v in fuzzy_measurements.items():
        if k in var_names:
            print(str(str(k) + ' = ' + str(v)))
            ctr_sys_sim.input[str(k)] = v

    ctr_sys_sim.compute()
    print('--- Simulation results ---')
    for vmfx in vmfx_list:
        if isinstance(vmfx, ctrl.Consequent):
            print(str(str(vmfx.label) + ' = ' + str(ctr_sys_sim.output[str(vmfx.label)])))
            target_consequent = vmfx
    return ctr_sys_sim, target_consequent


def view_defuzz(result_set, ctr_sys_sim):
    '''
    Plots the defuzzification results from the simulation.
    
        Args:
            result_set(np.array): the membership functions of the target variable
            ctr_sys_sim(ctrl.ControlSystemSimulation): the simulation object
            
    '''


    result_set.view(sim=ctr_sys_sim)
    plt.show()
