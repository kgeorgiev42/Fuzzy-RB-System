import os
import re
import numpy as np

def read_variables(file):
    with open(file) as fp:
        line = fp.readline()
        fuzzy_vars = {}
        while line:
            text = line.strip()
            text = str.split(text, " ")
            if len(text) == 1 and 'Rule' not in text[0] and text[0] != '':
                fuzzy_categories = {}
                var_name = text[0]
                fp.readline()
                line_cat = fp.readline()
                while len(line_cat.strip()) > 0:
                    category = line_cat.strip()
                    category_values = str.split(category, ' ')
                    cat_name = category_values[0].strip()
                    fuzzy_set = [float(category_values[1]), float(category_values[2]), float(category_values[3]), float(category_values[4])]
                    fuzzy_categories[str(cat_name).strip()] = eval(str(fuzzy_set))
                    line_cat = fp.readline()

                if len(fuzzy_categories) > 0:
                    fuzzy_vars[str(var_name)] = fuzzy_categories
            line = fp.readline()
        return fuzzy_vars

def read_rulebase(file, fuzzy_vars):
    with open(file) as fp:
        line = fp.readline()
        fuzzy_rules = []
        while line != "":
            if ':' in line:
                rule_text = str.split(line, ':')[1]
                if rule_text != "" and 'then' in rule_text:
                    precedent,result = str.split(rule_text, 'then')
                    fuzzy_rules_dict = {}
                    res_dict = {}
                    req_dict = {}
                    connector = 'SIMPLE'
                    if ' and ' in precedent:
                        connector = 'AND'
                        precedents = str.split(precedent, ' and ')
                    elif ' or ' in precedent:
                        connector = 'OR'
                        precedents = str.split(precedent, ' or ')
                    else:
                        precedents = precedent

                    for i in range(len(precedents)):
                        if i == 0:
                            if not isinstance(precedents, list):
                                precedents = str.split(precedents, 'If')[1]
                                match = re.search(r'(.*) is (.*)', precedents)
                                if match:
                                    if (match.groups()[0] and match.groups()[1]) is not None and match.groups()[0].strip() in fuzzy_vars:
                                        req = match.groups()[0].strip()
                                        outcome = match.groups()[1].strip()
                                        req_dict[str(req)] = outcome
                                break

                        match = re.search(r'(.*) is (.*)', precedents[i])
                        if match:
                            if (match.groups()[0] and match.groups()[1]) is not None:
                                if 'If' in match.groups()[0].strip():
                                    req = str.split(match.groups()[0].strip(), ' ')[1]
                                else:
                                    req = match.groups()[0].strip()
                                outcome = match.groups()[1].strip()
                                req_dict[str(req)] = outcome

                    result_lhs,result_rhs = str.split(result, 'is')
                    if (result_lhs and result_rhs) is not None and result_lhs.strip() in fuzzy_vars:
                        res_dict[str(result_lhs).strip()] = result_rhs.strip()

                    fuzzy_rules_dict['precedents'] = req_dict
                    fuzzy_rules_dict['connector'] = connector
                    fuzzy_rules_dict['result'] = res_dict
                    fuzzy_rules.append(fuzzy_rules_dict)

            line = fp.readline()

        return fuzzy_rules

def read_measurements(file, fuzzy_vars):
    with open(file) as fp:
        line = fp.readline()
        fuzzy_measurement_dict = {}
        while line:
            if '=' in line:
                variable,result = str.split(line, '=')
                if variable.strip() in fuzzy_vars:
                    fuzzy_measurement_dict[str(variable).strip()] = np.float32(result.strip())
            line = fp.readline()
        return fuzzy_measurement_dict

'''
fuzzy_variables = read_variables('dv.fuzzy')
print(fuzzy_variables)
fuzzy_rules = read_rulebase('dv.fuzzy', fuzzy_variables)
print(fuzzy_rules)
fuzzy_measurements = read_measurements('dv.fuzzy', fuzzy_variables)
print(fuzzy_measurements)
'''