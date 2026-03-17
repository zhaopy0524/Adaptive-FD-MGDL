# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 10:09:46 2024

@author: ZPY
"""
from singlegrade_dnn_main import single_dnn_main 


#set parameter
k = 50


#if use stochastic method in Adam, then SGD is 'True' and set minibatch size
#if use Full grade in Adam, the SGD is 'False' and set mini_batch size to 'False'
SGD = False                                          
# minibatch size
mini_batch_size = False


#set structure for SGDL
layers_dims = [2,256,256,256,256,256,1]                                                  # this is the structure for SGDL
#set train epoch
epochs = 8000


#set max learning rate and min learning rate 
max_learning_rate = 1e-2                                            # the maximum learning rate, denote as t_max in the paper
min_learning_rate = 1e-4                                            # the minimum learning rate, denote as t_min in the paper


single_dnn_main( layers_dims, max_learning_rate, min_learning_rate, epochs, mini_batch_size, SGD, k )
