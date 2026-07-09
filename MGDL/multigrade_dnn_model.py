# -*- coding: utf-8 -*-

import multigrade_dnn_solving as m_dnn


def multigrade_dnn_model(data, nn_parameter, opt_parameter, trained_variable):
        
    """
    implement a multigrade linear composition model 
    
    Parameters
    ----------
    data :              dictionary 
                        the information of orginal data  (train_X, train_Y, test_X, test_Y)          
    nn_parameter :      dictionary
                        the information of model (structure of network, regularization parameters)
    opt_parameter :     dictionary
                        the information of optimization 
                        containing (optimizer,  learning_rate, mini_batch_size, beta1, beta2, epsilon, error, epochs)
    trained_variable :  dictionary 
                        pretrained for fixed information 
        
    Returns
    -------
    trained_variable :  dictionary
                        updated pretrained for fixed information 
        
    """                                 
    grade_length = len(nn_parameter['mul_layers_dims'])           # the length of new grade will be trained

    
    #trained the new layer
    for i in range(1,  grade_length + 1):
        print("\n----------------------grade : {}---------------------\n".format(i))

        layers_dims = nn_parameter["mul_layers_dims"][i-1]
        max_learning_rate = opt_parameter["MAX_learning_rate"][i-1]
        min_learning_rate = opt_parameter["MIN_learning_rate"][i-1]
        epochs = opt_parameter["epochs"][i-1]
        activation = nn_parameter["activation"][i-1]
        stop_criterion = opt_parameter["Stop_criterion"][i-1]

        if i==1:
            trained_variable, prev_info = m_dnn.multigrade_dnn_model_grade_1(trained_variable, data, layers_dims, opt_parameter, stop_criterion,
                                                                             max_learning_rate, min_learning_rate, epochs, activation)
        else:           
            trained_variable, prev_info = m_dnn.multigrade_dnn_model_grade_ell(trained_variable, data, prev_info, layers_dims, opt_parameter, stop_criterion, 
                                                                               max_learning_rate, min_learning_rate, epochs, activation)

            
    return trained_variable, prev_info


      
        
def multigrade_dnn_model_predict(data, nn_parameter, opt_parameter, trained_variable):
    """
    preict for test data after the parameter are trained
    
    Parameters
    ----------
    data :              dictionary 
                        the information of orginal data  (train_X, train_Y, test_X, test_Y)          
    nn_parameter :      dictionary
                        the information of model (structure of network, regularization parameters)
    opt_parameter :     dictionary
                        the information of optimization 
                        containing (optimizer,  learning_rate, mini_batch_size, beta1, beta2, epsilon, error, epochs)
    trained_variable :  dictionary 
                        pretrained for fixed information     
    
    
    
    
    """
    
    
    grade_length = len(nn_parameter['mul_layers_dims']) 
    
    test_rse = []
    mul_predict_test_Y = []
    
    
    for i in range(1,  grade_length + 1):
        layers_dims = nn_parameter["mul_layers_dims"][i-1]
        parameters = trained_variable['mul_parameters'][i-1]
        activation = nn_parameter["activation"][i-1]
        
        if i==1:
            test_N, test_caches = m_dnn.multigrade_model_forward(data['test_X'], layers_dims, parameters, activation)
            test_N_prev_X = test_caches[-1][0][0]  
            predict_test_Y = test_N
            rse = m_dnn.rse(data['test_Y'], predict_test_Y)
            test_rse.append(rse)
            
        else:
            test_N, test_caches = m_dnn.multigrade_model_forward(test_N_prev_X, layers_dims, parameters, activation)
            test_N_prev_X = test_caches[-1][0][0]  
            predict_test_Y = test_N + predict_test_Y
            rse = m_dnn.rse(data['test_Y'], predict_test_Y)
            test_rse.append(rse)
            
        mul_predict_test_Y.append(predict_test_Y)
            
            
    return test_rse, mul_predict_test_Y
