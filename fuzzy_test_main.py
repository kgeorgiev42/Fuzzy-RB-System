
from modules.fuzzy_membership import create_membership_functions, plot_fuzzy_sets
from modules.fuzzy_inference import map_variable_types, infer_rules
from modules.fuzzy_defuzzifier import defuzzify_bisector, defuzzify_centroid, plot_defuzz
import sys


if __name__ == '__main__':
    fuzzy_dict, x_ranges, var_names, fuzzy_variables = create_membership_functions(sys.argv[1])
    plot_fuzzy_sets(fuzzy_dict, x_ranges)

    vmfx_list, fuzzy_measurements = map_variable_types(sys.argv[1], fuzzy_variables, var_names, x_ranges, fuzzy_dict)

    activation_dict = infer_rules(sys.argv[1], fuzzy_variables, fuzzy_dict, fuzzy_measurements, x_ranges)
    c_res,c_x,c_mfx = defuzzify_centroid(activation_dict, vmfx_list)
    b_res,b_x,b_mfx = defuzzify_bisector(activation_dict, vmfx_list)
    plot_defuzz(vmfx_list,fuzzy_dict,c_res,c_x,c_mfx,b_res,b_x,b_mfx)
