# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import pickle
import numpy as np
import re
import multigrade_dnn_model as m_dnn
import pandas as pd

from multigrade_FD import generate_data

mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['axes.edgecolor'] = 'black'
mpl.rcParams['axes.grid'] = False
mpl.rcParams['grid.color'] = 'white'
mpl.rcParams['grid.linestyle'] = '-'
mpl.rcParams['grid.linewidth'] = 1.0
mpl.rcParams['legend.facecolor'] = 'white'
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['axes.spines.top'] = True
mpl.rcParams['axes.spines.right'] = True
mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.bottom'] = True
mpl.rcParams['xtick.bottom'] = True
mpl.rcParams['ytick.left'] = True
mpl.rcParams['figure.autolayout'] = True

def format_with_comma(x, pos):
    return f"{int(x):,}"

# fullfilename = 

match = re.search(r"k=(\d+)", fullfilename)
k = int(match.group(1))

with open(fullfilename, 'rb') as f:
    trained_variable, nn_parameter, opt_parameter, prev_info = pickle.load(f)

print("################################################################################################################")
print(fullfilename)
print(nn_parameter)
print(opt_parameter)

data = generate_data(k)

test_rse, predict_test_Y = m_dnn.multigrade_dnn_model_predict(data, nn_parameter, opt_parameter, trained_variable)

num_iter = trained_variable["REC_FRQ_iter"]

train_rse = []
TRAIN_loss = []
MUL_EPOCH = []

total_time = 0

for i in range(0, len(nn_parameter["mul_layers_dims"])):
    total_time = total_time + trained_variable['train_time'][i]
    train_rse.append(trained_variable['train_rse'][i][-1])
    if i == 0:
        current_epoch = trained_variable["REC_FRQ_iter"][i][-1]
        MUL_EPOCH.append(current_epoch)
        TRAIN_loss.extend(trained_variable['train_losss'][i])
    else:
        current_epoch += trained_variable["REC_FRQ_iter"][i][-1]
        MUL_EPOCH.append(current_epoch)
        TRAIN_loss.extend(trained_variable['train_losss'][i][1:])

print('the train rse for each grade is {}'.format(train_rse))
print('the test rse for each grade is  {}'.format(test_rse))
print('the train times for each grade is {}'.format(trained_variable['train_time']))
print('the total train times is {}'.format(total_time))


plt.plot(np.array(TRAIN_loss), label='Train loss')
plt.legend(loc='upper right')
plt.xlabel("Number of training epochs")
plt.ylabel("Loss")
plt.yscale('log')
plt.xlim([0, MUL_EPOCH[0]])

plt.title('MGDL: Loss ($k$={})'.format(k))
for i in range(0, len(nn_parameter["mul_layers_dims"]) - 1):
    plt.axvline(x=MUL_EPOCH[i], color='k', linestyle=':')
plt.xticks([0] + [MUL_EPOCH[i] for i in range(len(MUL_EPOCH))], rotation=45)

ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_with_comma))

plt.tight_layout()
plt.grid(True)

plt.show()




# for i in range(0, len(nn_parameter["mul_layers_dims"])):
#     num_iter = trained_variable["REC_FRQ_iter"][i]

#     X1 = data['test_X1']
#     X2 = data['test_X2']
#     n = data['ntest']
#     Z = predict_test_Y[i][0].reshape(n, n)

#     plt.figure(figsize=(8, 6))
#     plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
#     plt.colorbar(label='$u(x_1, x_2)$')
#     plt.xlabel('$x_1$')
#     plt.ylabel('$x_2$')
#     plt.title('MGDL: grade {} ($k$={})'.format(i + 1, k))

#     # scatter_filename = "grade{}.png".format(i + 1)
#     # plt.savefig(scatter_filename, format="png")

#     # print(f"Scatter plot for grade {i + 1} saved as {scatter_filename}")
#     plt.show()
    
# z1 =Z
# df = pd.DataFrame(Z)
# df.to_excel("u_test.xlsx", index=False, header=False)


# Z = data['test_Y'][0].reshape(n, n)
# plt.figure(figsize=(8, 6))
# plt.contourf(X1, X2, z1-Z, levels=50, cmap='viridis')
# plt.colorbar(label='$u(x_1, x_2)$')
# plt.xlabel('$x_1$')
# plt.ylabel('$x_2$')
# plt.title('Exact solution')

# scatter_filename = "exact.png"
# plt.savefig(scatter_filename, format="png")
# plt.show()




