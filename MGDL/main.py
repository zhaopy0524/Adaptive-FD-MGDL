# -*- coding: utf-8 -*-


from multigrade_dnn_main import multi_grade_dnn
from multigrade_FD import generate_data

#set parameter

SGD = False
mini_batch = False

k = 50

#set structure for MGDL
mul_layers_dims =  [[2, 256, 256, 1]]                  # this is the structure for MGDL
#set activation for MGDL for each grade
activation = ['sin']
#set train epoch for each grade
mul_epochs = [500]

stop_criterion = [1e-06]

data = generate_data(k)

max_learning_rate = [0.1]
min_learning_rate = [0.0001]
multi_grade_dnn(data, stop_criterion, max_learning_rate, min_learning_rate, mul_layers_dims, mul_epochs, SGD, mini_batch, activation, k)