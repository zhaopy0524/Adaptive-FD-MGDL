# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 10:02:39 2024

@author: ZPY
"""

import pickle
import os.path
import singlegrade_dnn_solving as dnn
from singlegrade_FD import generate_data



def single_dnn_main(layers_dims, max_learning_rate, min_learning_rate, epochs, mini_batch_size, SGD, k):

    #---------neural network parameter--
    nn_parameter = {}   
    nn_parameter["layers_dims"] = layers_dims
    nn_parameter["lambd_W"] = 0
    nn_parameter["sinORrelu"] = 3
    nn_parameter["activation"] = "relu"
    nn_parameter["init_method"] = "xavier"
    #-----------------------------------
    
    
    
    #-------optimization parameter--
    opt_parameter = {}
    opt_parameter["optimizer"] = "adam"
    opt_parameter["beta1"] = 0.9
    opt_parameter["beta2"] = 0.999
    opt_parameter["epsilon"] = 1e-8
    opt_parameter["error"] = 1e-7
    opt_parameter["max_learning_rate"] = max_learning_rate
    opt_parameter["min_learning_rate"] = min_learning_rate
    opt_parameter["epochs"] = epochs
    opt_parameter["REC_FRQ"] = 1
    opt_parameter["SGD"] = SGD
    opt_parameter["mini_batch_size"] = mini_batch_size
    #-----------------------------

    data = generate_data(k)

    history = dnn.singlegrade_dnn_model_grade(data, nn_parameter, opt_parameter)

    save_path = 'k={}'.format(k) 
    os.makedirs(save_path, exist_ok=True)

    filename = "SGDL_xavier_epoch{}_minibatch{}_MAXlr{:.4e}_MINlr{:.4e}_train{:.4e}.pickle".format(
             epochs, 
             opt_parameter["mini_batch_size"],
             opt_parameter["max_learning_rate"],
             opt_parameter["min_learning_rate"],
             history['train_rses'][-1]  )
    fullfilename = os.path.join(save_path, filename)  
    
    with open(fullfilename, 'wb') as f:
        pickle.dump([history, nn_parameter, opt_parameter],f)
        
    print(f"文件已保存到 {fullfilename}")