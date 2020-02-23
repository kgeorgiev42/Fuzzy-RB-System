import seaborn as sns
import matplotlib.pyplot as plt

from modules.fuzzy_load import *

sns.set(style='darkgrid', palette="Paired")


def create_membership_functions(file, from_file=True):
    '''
    Creates trapezoidal membership functions using the parsed fuzzy variables.
    Generates discrete values of membership based on the estimated range of the variables.

        Args:
            file(str): the input filename
            from_file: feed the file as input or feed an already parsed dictionary (def. True)

        Returns:
            fuzzy_dict(dict): the processed fuzzy variable dictionary with assigned memberships
            x_ranges(dict): the generated range of values for each membership
            var_names(list): list of variable names for lookup
            fuzzy_variables(dict): the original parsed fuzzy variable dictionary
            
    '''


    if from_file:
        fuzzy_variables = read_variables(file)
    else:
        fuzzy_variables = file
    fuzzy_dict = {}
    x_ranges = {}
    var_names = []
    for k, v in fuzzy_variables.items():
        var_name = k
        max_range = 0
        fuzzy_val_dict = {}
        for k_j, v_j in v.items():
            max_range_j = v_j[1] + v_j[3]
            if max_range_j > max_range:
                max_range = max_range_j

        x_range = np.arange(0, max_range + 0.1, 0.1)

        for k_j, v_j in v.items():
            cat_name = k_j
            a, b, c, d = np.r_[np.float32([v_j[0] - v_j[2], v_j[0], v_j[1], v_j[3] + v_j[1]])]
            y = np.ones(len(x_range))

            ### triangle membership 1
            idx = np.nonzero(x_range <= b)[0]

            a1, b1, c1 = np.r_[np.r_[a, b, b]]
            y1 = np.zeros(len(x_range[idx]))

            # Left side
            if a1 != b1:
                idx1 = np.nonzero(np.logical_and(a1 < x_range[idx], x_range[idx] < b1))[0]
                y1[idx1] = (x_range[idx][idx1] - a1) / float(b1 - a1)

            # Right side
            if b1 != c1:
                idx1 = np.nonzero(np.logical_and(b1 < x_range[idx], x_range[idx] < c1))[0]
                y1[idx1] = (c1 - x_range[idx][idx1]) / float(c1 - b1)

            idx1 = np.nonzero(x_range[idx] == b1)
            y1[idx1] = 1
            y[idx] = y1

            ### Triangle membership 2
            idx = np.nonzero(x_range >= c)[0]

            a2, b2, c2 = np.r_[np.r_[c, c, d]]
            y2 = np.zeros(len(x_range[idx]))

            # Left side
            if a2 != b2:
                idx2 = np.nonzero(np.logical_and(a2 < x_range[idx], x_range[idx] < b2))[0]
                y2[idx2] = (x_range[idx][idx2] - a2) / float(b2 - a2)

            # Right side
            if b2 != c2:
                idx2 = np.nonzero(np.logical_and(b2 < x_range[idx], x_range[idx] < c2))[0]
                y2[idx2] = (c2 - x_range[idx][idx2]) / float(c2 - b2)

            idx2 = np.nonzero(x_range[idx] == b2)
            y2[idx2] = 1
            y[idx] = y2

            idx = np.nonzero(x_range < a)[0]
            y[idx] = np.zeros(len(idx))

            idx = np.nonzero(x_range > d)[0]
            y[idx] = np.zeros(len(idx))

            fuzzy_val_dict[str(cat_name)] = y

        x_ranges[str(var_name)] = x_range
        fuzzy_dict[str(var_name)] = fuzzy_val_dict
        var_names.append(var_name)

    return fuzzy_dict, x_ranges, var_names, fuzzy_variables


def plot_fuzzy_sets(fuzzy_dict, x_ranges):
    '''
    Creates one plot for each fuzzy variable and displays the resulting sets.
    
        Args:
            fuzzy_dict(dict): the processed fuzzy variable dictionary with assigned memberships
            x_ranges(dict): the generated range of values for each membership
            
    '''


    for k, v in fuzzy_dict.items():
        var_name = k
        plt.figure(figsize=(8, 6))
        plt.title(str(var_name))
        for k_j, v_j in v.items():
            sns.lineplot(x_ranges[var_name], v_j, label=str(k_j), linewidth=3)
            plt.ylabel('Fuzzy membership value')
            plt.xlabel('Variables')
            plt.ylim(-0.01, 1.1)
            plt.legend()
        plt.show()
