# -*- coding: utf-8 -*-


import pickle
import re
import os.path
import multigrade_dnn_solving as m_dnn

from multigrade_FD import generate_data

def extract_train_value(filepath):
    match = re.search(r'train([\d\.eE+-]+)', filepath)
    if match:
        value_str = match.group(1).rstrip('.')
        return float(value_str)
    return float('inf')

def add_network(fullfilename):
    match = re.search(r"k=(\d+)", fullfilename)
    k = int(match.group(1))
    data = generate_data(k)

    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # add new grade

    MAXLr = [1e-1,1e-2,1e-3]
    MINLr = [1e-1,1e-2,1e-3,1e-4]

    #4
    saved_file_paths = []
    for i in range(0,3):
        for j in range(i,4):

            with open(fullfilename, 'rb') as f:
                trained_variable, nn_parameter, opt_parameter, prev_info = pickle.load(f)

            grade_length = len(nn_parameter['mul_layers_dims'])           # the length of new grade will be trained

            nn_parameter["mul_layers_dims"].append([16, 16, 1])
            opt_parameter["epochs"].append(2000)
            opt_parameter["Stop_criterion"].append(1e-8)
            nn_parameter["activation"].append("relu")

            opt_parameter["MAX_learning_rate"].append(MAXLr[i])
            opt_parameter["MIN_learning_rate"].append(MINLr[j])

            print("\n----------------------grade : {}---------------------\n".format(grade_length + 1))
            layers_dims = nn_parameter["mul_layers_dims"][grade_length]
            max_learning_rate = opt_parameter["MAX_learning_rate"][grade_length]
            min_learning_rate = opt_parameter["MIN_learning_rate"][grade_length]
            epochs = opt_parameter["epochs"][grade_length]
            activation = nn_parameter["activation"][grade_length]
            stop_criterion = opt_parameter["Stop_criterion"][grade_length]

            trained_variable, prev_info = m_dnn.multigrade_dnn_model_grade_ell(trained_variable, data, prev_info, layers_dims, opt_parameter, stop_criterion,
                                                                                    max_learning_rate, min_learning_rate, epochs, activation)

            # Construct a preservation path
            save_path = 'k={}'.format(k)
            filename = "MGDL_xavier_epoch{}_MAXlr{:.4e}_MINlr{:.4e}_train{:.4e}.pickle".format(
                      opt_parameter["epochs"],
                      opt_parameter["MAX_learning_rate"][grade_length],
                      opt_parameter["MIN_learning_rate"][grade_length],
                      trained_variable['train_rse'][-1][-1])
            fullfilename = os.path.join(save_path, filename)

            saved_file_paths.append(fullfilename)

            # Ensure that the directory of the save path exists
            os.makedirs(save_path, exist_ok=True)

            # Save the file
            with open(fullfilename, 'wb') as f:
                pickle.dump([trained_variable, nn_parameter, opt_parameter, prev_info], f)


            opt_parameter["MAX_learning_rate"].pop()
            opt_parameter["MIN_learning_rate"].pop()
    return

if __name__ == "__main__":
    # fullfilename = 
    add_network(fullfilename)