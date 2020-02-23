import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style='darkgrid', palette="Paired")


def defuzzify_centroid(activation_dict, vmfx_list):
    '''
    Estimates the defuzzified value using the centroid method.
    The membership activations are aggregated based on their maximum value within the range.
    This area/centroid estimator is largely based on scikit-fuzzy's centroid method:https://github.com/scikit-fuzzy/scikit-fuzzy/blob/master/skfuzzy/defuzzify/defuzz.py

        Args:
            activation_dict(dict): membership activation values throughout the range
            vmfx_list(list): list of dictionaries containing the variable names, ranges and membership functions

        Returns:
            result(np.float32): estimated defuzzified value
            conseq_range(np.array): the range of possible values for the consequent variable aggregated_mfx(np.array): the aggregated activation functions for each rule
            
    '''


    aggregated_mfx = np.zeros_like(list(activation_dict.values())[0])

    for i in range(len(aggregated_mfx)):
        max_val = 0.0
        for k, v in activation_dict.items():
            if v[i] > max_val:
                max_val = v[i]
        aggregated_mfx[i] = max_val

    for vmfx in vmfx_list:
        if vmfx['type'] == 'Consequent':
            conseq_range = vmfx['range']
            conseq_name = vmfx['name']

    sum_centroid_area = 0.0
    sum_area = 0.0
    result = 0.0

    if conseq_range is not None:

        # If the membership function is a singleton fuzzy set:
        if len(conseq_range) == 1:
            result = conseq_range[0] * aggregated_mfx[0] / np.float32(aggregated_mfx[0])

        else:
            # else return the sum of centroid*area/sum of area
            for i in range(1, len(conseq_range)):
                x1 = conseq_range[i - 1]
                x2 = conseq_range[i]
                y1 = aggregated_mfx[i - 1]
                y2 = aggregated_mfx[i]

                # if y1 == y2 == 0.0 or x1==x2: --> rectangle of zero height or width
                if not (y1 == y2 == 0.0 or x1 == x2):
                    if y1 == y2:  # rectangle
                        centroid = 0.5 * (x1 + x2)
                        area = (x2 - x1) * y1
                    elif y1 == 0.0 and y2 != 0.0:  # triangle, height y2
                        centroid = 2.0 / 3.0 * (x2 - x1) + x1
                        area = 0.5 * (x2 - x1) * y2
                    elif y2 == 0.0 and y1 != 0.0:  # triangle, height y1
                        centroid = 1.0 / 3.0 * (x2 - x1) + x1
                        area = 0.5 * (x2 - x1) * y1
                    else:
                        centroid = (2.0 / 3.0 * (x2 - x1) * (y2 + 0.5 * y1)) / (y1 + y2) + x1
                        area = 0.5 * (x2 - x1) * (y1 + y2)

                    sum_centroid_area += centroid * area
                    sum_area += area

            result = np.round(sum_centroid_area / np.float32(sum_area), 2)

    print('Centroid defuzzified value for {}:{}'.format(conseq_name, result))
    return result, conseq_range, aggregated_mfx


def defuzzify_bisector(activation_dict, vmfx_list):
    '''
    Estimates the defuzzified value using the bisector method.
    The membership activations are aggregated based on their maximum value within the range.
    This area/subarea estimator is largely based on scikit-fuzzy's bisector method:https://github.com/scikit-fuzzy/scikit-fuzzy/blob/master/skfuzzy/defuzzify/defuzz.py

        Args:
            activation_dict(dict): membership activation values throughout the range
            vmfx_list(list): list of dictionaries containing the variable names, ranges and membership functions

        Returns:
            result(np.float32): estimated defuzzified value
            conseq_range(np.array): the range of possible values for the consequent variable aggregated_mfx(np.array): the aggregated activation functions for each rule
            
    '''


    aggregated_mfx = np.zeros_like(list(activation_dict.values())[0])

    for i in range(len(aggregated_mfx)):
        max_val = 0.0
        for k, v in activation_dict.items():
            if v[i] > max_val:
                max_val = v[i]
        aggregated_mfx[i] = max_val

    for vmfx in vmfx_list:
        if vmfx['type'] == 'Consequent':
            conseq_range = vmfx['range']
            conseq_name = vmfx['name']

    sum_area = 0.0
    acc_area = [0.0] * (len(conseq_range) - 1)
    result = 0.0

    if conseq_range is not None:

        # If the membership function is a singleton fuzzy set:
        if len(conseq_range) == 1:
            result = conseq_range[0]

        else:
            # else return the sum of centroid*area/sum of area
            for i in range(1, len(conseq_range)):
                x1 = conseq_range[i - 1]
                x2 = conseq_range[i]
                y1 = aggregated_mfx[i - 1]
                y2 = aggregated_mfx[i]

                # if y1 == y2 == 0.0 or x1==x2: --> rectangle of zero height or width
                if not (y1 == y2 == 0.0 or x1 == x2):
                    if y1 == y2:  # rectangle
                        area = (x2 - x1) * y1
                    elif y1 == 0.0 and y2 != 0.0:  # triangle, height y2
                        area = 0.5 * (x2 - x1) * y2
                    elif y2 == 0.0 and y1 != 0.0:  # triangle, height y1
                        area = 0.5 * (x2 - x1) * y1
                    else:
                        area = 0.5 * (x2 - x1) * (y1 + y2)

                    sum_area += area
                    acc_area[i - 1] = sum_area

            index = np.nonzero(np.array(acc_area) >= sum_area / 2.0)[0][0]

            if index == 0:
                subarea = 0
            else:
                subarea = acc_area[index - 1]

            x1 = conseq_range[index]
            x2 = conseq_range[index + 1]
            y1 = aggregated_mfx[index]
            y2 = aggregated_mfx[index + 1]

            subarea = sum_area / 2.0 - subarea

            diff = x2 - x1
            if y1 == y2:  # rectangle
                result = np.round(subarea / y1 + x1, 2)
            elif y1 == 0.0 and y2 != 0.0:  # triangle, height y2
                root = np.sqrt(2.0 * subarea * diff / y2)
                result = np.round((x1 + root), 2)
            elif y2 == 0.0 and y1 != 0.0:  # triangle, height y1
                root = np.sqrt(diff * diff - (2.0 * subarea * diff / y1))
                result = np.round((x2 - root), 2)
            else:                          # trapezium
                m = (y2 - y1) / diff
                root = np.sqrt(y1 * y1 + 2.0 * m * subarea)
                result = np.round((x1 - (y1 - root) / m), 2)

    print('Bisector defuzzified value for {}:{}'.format(conseq_name, result))
    return result, conseq_range, aggregated_mfx


def plot_defuzz(vmfx_list, fuzzy_dict, c_res, c_x, c_mfx, b_res, b_x, b_mfx):
    '''
    Plots the defuzzification results of the centroid and bisector methods.
    
        Args:
            vmfx_list(list): list of dictionaries containing the variable names, ranges and membership functions 
            fuzzy_dict(dict): the original parsed fuzzy variable dictionary
            c_res(np.float32): estimated defuzzified value using the centroid method
            c_x(np.array): the range of possible values for the consequent variable (returned from the centroid method)
            c_mfx(np.array): the aggregated activation functions for each rule (returned from the centroid method)
            b_res(np.float32): estimated defuzzified value using the bisector method
            b_x(np.array): the range of possible values for the consequent variable (returned from the bisector method)
            b_mfx(np.array): the aggregated activation functions for each rule (returned from the bisector method)

            
    '''


    tip0 = np.zeros_like(c_x)
    for vmfx in vmfx_list:
        if vmfx['type'] == 'Consequent':
            conseq_name = vmfx['name']

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['r', 'b', 'g', 'c', 'v']
    c_idx = 0

    for k, v in fuzzy_dict.items():
        if str(k) == conseq_name:
            for k_j, v_j in v.items():
                c_idx += 1
                if c_idx == len(colors) - 1:
                    c_idx = 0
                ax.plot(c_x, v_j, c_idx, linewidth=2, linestyle='--')
                ax.set_ylim(-0.01, 1.1)

    ax.fill_between(c_x, tip0, c_mfx, facecolor='Cyan', alpha=0.3, label='Area of membership')
    c_activation = np.interp(c_res, c_x, c_mfx, left=0, right=0)
    b_activation = np.interp(b_res, b_x, b_mfx, left=0, right=0)
    ax.plot([c_res, c_res], [0, c_activation], 'k', linewidth=1.5, alpha=0.9, color='darkgreen', label='CDV')
    ax.plot([b_res, b_res], [0, b_activation], 'k', linewidth=1.5, alpha=0.9, color='darkred', label='BDV')
    ax.set_title('Defuzzification results for Centroid and Bisector methods')
    plt.legend()
    plt.ylabel('Fuzzy membership value')
    plt.xlabel('Variables')
    plt.show()
