# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 10:11:18 2024

@author: ZPY
"""

import re
import pickle
import pandas as pd
import numpy as np
import matplotlib as mpl
import singlegrade_dnn_solving as dnn
from singlegrade_FD import generate_data

# 设置全局样式
background_color = '#EAEAF2'
mpl.rcParams['axes.facecolor'] = background_color  # 图表背景色
mpl.rcParams['axes.edgecolor'] = background_color  # 边框颜色
mpl.rcParams['axes.grid'] = True  # 显示网格
mpl.rcParams['grid.color'] = 'white'  # 网格线颜色
mpl.rcParams['grid.linestyle'] = '-'  # 网格线样式
mpl.rcParams['grid.linewidth'] = 1.0
mpl.rcParams['legend.facecolor'] = background_color  # 图例背景色
mpl.rcParams['xtick.color'] = 'black'  # x轴刻度颜色
mpl.rcParams['ytick.color'] = 'black'  # y轴刻度颜色
mpl.rcParams['axes.spines.top'] = False  # 隐藏顶部边框
mpl.rcParams['axes.spines.right'] = False  # 隐藏右侧边框
mpl.rcParams['axes.spines.left'] = False  # 隐藏左侧边框
mpl.rcParams['axes.spines.bottom'] = False  # 隐藏底部边框
mpl.rcParams['xtick.bottom'] = False  # 隐藏 x 轴刻度线
mpl.rcParams['ytick.left'] = False  # 隐藏 y 轴刻度线
mpl.rcParams['figure.autolayout'] = True  # 自动布局


# fullfilename =

match = re.search(r"k=(\d+)", fullfilename)
k = int(match.group(1))
data = generate_data(k)

with open(fullfilename, 'rb') as f:
    history, nn_parameter, opt_parameter = pickle.load(f)


train_loss =  np.array(history["train_costs"])

df = pd.DataFrame(train_loss, columns=["train_loss"])
# df = pd.DataFrame(history["train_costs"], columns=["train_loss"])
df.to_excel("train_losss.xlsx", index=False)


# train_predict, _ = singlegrade_model_forward(data["train_X"], nn_parameter['layers_dims'], history['parameters'], nn_parameter["activation"] , nn_parameter["sinORrelu"])
test_predict, _ = dnn.singlegrade_model_forward(data["test_X"], nn_parameter['layers_dims'], history['parameters'], nn_parameter["activation"] , nn_parameter["sinORrelu"])
u_test = test_predict.reshape(data['ntest'],data['ntest'])


print("###########################################################################")
print(fullfilename)
print(nn_parameter)
print(opt_parameter)

print('train_rse is {}, test_rse is {}'.format(history['train_rses'][-1], dnn.rse(data["test_Y"], test_predict)))

print('the train time is {}'.format(history["time"]))  