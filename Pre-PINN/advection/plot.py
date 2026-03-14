import math

import torch
from torch.func import hessian
from matplotlib import pyplot as plt


def plot_hessian(fn, net_args, argnums=0, title=''):
    hess = hessian(fn, argnums=argnums)(*net_args)

    params = net_args[argnums]
    his = []
    for hi, pi in zip(hess, params):
        hijs = []
        for hij in hi:
            hij_flat = hij.reshape(*pi.shape, -1)
            hij_flat = hij_flat.reshape(-1, hij_flat.shape[-1])
            hijs.append(hij_flat)
        hi = torch.cat(hijs, 1)
        his.append(hi)
    hess_flat = torch.cat(his, 0)

    # alternative method to flatten
    # precond_pde_loss_flat = lambda v: fn(fnet, v2p(v, params), buffers, data)
    # hess = hessian(precond_pde_loss_flat, argnums=0)(p2v(params))

    eigenvalues = torch.linalg.eigvalsh(hess_flat, UPLO='U')
    eigenvalues_np = eigenvalues.detach().cpu().numpy()
    plt.hist(eigenvalues_np, bins=100, alpha=0.7, color='blue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.show()
    plt.title(title)
    plt.imshow(hess_flat.detach().cpu())
    plt.colorbar()
    plt.show()