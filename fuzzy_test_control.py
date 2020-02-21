from modules.fuzzy_membership import create_membership_functions, plot_fuzzy_sets
from modules.fuzzy_control_system import map_variable_types, create_rule_control_system, apply_rules, view_defuzz
import sys

if __name__ == '__main__':
    fuzzy_dict, x_ranges, var_names, fuzzy_variables = create_membership_functions(sys.argv[1])
    plot_fuzzy_sets(fuzzy_dict, x_ranges)
    vmfx_list, fuzzy_measurements = map_variable_types(sys.argv[1], fuzzy_variables, var_names, x_ranges, fuzzy_dict)
    #view_sample_set(vmfx_list[2], 'average')
    rcs = create_rule_control_system(sys.argv[1], fuzzy_variables, var_names, vmfx_list)
    #plot_rule_graphs(rcs)

    ctr_sys_sim, consequent = apply_rules(rcs, fuzzy_measurements, var_names, vmfx_list)
    view_defuzz(consequent, ctr_sys_sim)