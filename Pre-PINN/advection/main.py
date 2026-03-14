from typing import Iterable

from matplotlib import pyplot as plt
import warnings
import matplotlib
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

import argparse
import os
from functools import partial

import scipy.optimize
import torch
import torchopt
from torch import vmap

from save_utils import save
from utils import compute_empirical_A_mc, make_functional_with_buffers
from plot import plot_hessian
from precond import Preconditioner
from net import create_net
from grid import Grid, Sobolev_sampling
from pde import PDE
from bcs import BoundaryConditions


def losses(pde, bcs, lmbda, fnet, params, buffers, data):
    def fnet_single(params, buffers, x):
        return fnet(params, buffers, x.unsqueeze(0)).squeeze(0)
    res_pde_op = vmap(pde.residual_single, in_dims=(None, None, None, None, 0))
    res_pde = res_pde_op(True, fnet_single, params, buffers, data.x_interior)
    res_bcs = bcs.residuals(True, fnet_single, params, buffers, data.x_boundaries, norm = True)
    weighted_bcs = (lmbda * torch.sum(res ** 2) for res in res_bcs)
    volume = torch.prod(data.extent)
    return volume*(res_pde ** 2).sum()/res_pde.shape[0], sum(weighted_bcs)


def step(optimizer, opt_state, precond, losses_fn, fnet, params, buffers, data):
    loss_res, loss_bcs = losses_fn(fnet, params, buffers, data)
    grad = torch.autograd.grad(loss_res + loss_bcs, params)  # compute gradients
    if precond.on == 'grads':
        grad = precond(grad, 'full')
    updates, opt_state = optimizer.update(grad, opt_state)  # get updates
    params = torchopt.apply_updates(params, updates, inplace=False)  # update network parameters
    return loss_res.item(), loss_bcs.item(), params, opt_state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pde', type=str, default='poisson', choices=['helmholtz', 'poisson', 'advection'])
    parser.add_argument('--bcs', type=str, default='zero_bc', choices=['mixed_left', 'zero_bc', 'advection_2d'])
    parser.add_argument('--device', type=str, default='cuda:0', choices=['cuda:0', 'cuda:1', 'cpu'], help='Used device')
    parser.add_argument('--res', nargs='+', help='Resolution', default=1000, type=int)
    parser.add_argument('--nfreqs', nargs='+', type=int, default=16, help='number of frequencies in the fourier nets')
    # parser.add_argument('--res', type=int, default=1000, help='Domain resolution')
    parser.add_argument('--nepochs', type=int, default=200)
    parser.add_argument('--net', type=str, default='deep_fourier_basis_1d', choices=['mlp', 'fourier_basis', 'deep_fourier_basis_1d', 'learnable_fourier_feats_1d'])
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='Optimizer of choice.')
    parser.add_argument('--lr', default=None, help='Learning rate. If None, it assume you use golden search with lr=1/lambda_max')
    parser.add_argument('--precond', type=str, default='poisson_fourier_basis_1d', choices=['none', 'helmholtz_fourier_basis_1d', 'poisson_fourier_basis_1d', 'advection_2d'])
    parser.add_argument('--precond_on', type=str, default='grads', choices=['grads', 'params', 'net'])
    parser.add_argument('--lmbda', default="use_golden_search", help="List of lambda values. If lambdas=optimize, it will find them with golden search")
    parser.add_argument('--output', type=str, default='results/test/', help='Output folder for saving results and config.')
    parser.add_argument('--pde_param', type=int, default=6, help='parameter of the pde, e.g. resonance frequency, etc')
    args = parser.parse_args()
    print(args)

    torch.manual_seed(0)
    dim = len(args.res) if isinstance(args.res, Iterable) else 1
    extent = [2*torch.pi, 1.][:dim]

    print('Resolution', args.res)
    data = Grid(extent, args.res, args.device)

    # seed = 3
    # log2_res = torch.tensor([8, 6])
    # sampling = Sobolev_sampling(extent, log2_res, args.device, seed)

    pde = PDE(args.pde, args.pde_param)
    bcs = BoundaryConditions(args.bcs, extent)
    net = create_net(args, data).to(args.device)
    fnet_no_precond, params, buffers = make_functional_with_buffers(net)
    precond = Preconditioner(args, data).to(args.device)
    if args.precond_on == 'params':
        fnet = lambda params, buffers, x: fnet_no_precond(precond(params, 'half'), buffers, x)
    else:
        fnet = fnet_no_precond

    # plt.title('output at epoch 0')
    # plt.plot(fnet(params, buffers, data.x_interior).detach().cpu())
    # plt.show()

    if args.lmbda == 'use_golden_search':
        print('Optimizing for lambdas!')
        assert args.lr is None, 'you cannot optimize and choose a lr at the same time'
        data_fd = Grid(extent, torch.tensor([256, 2048]), args.device)
        lmbda2cn = lambda lmbda: compute_empirical_A_mc(pde, bcs, precond, lmbda, fnet, params, buffers, data_fd, fd = True)[0].item()
        positive_lam2cn = lambda x: float('inf') if x < 1e-6 else lmbda2cn(x)
        lmbda = scipy.optimize.golden(positive_lam2cn, maxiter=10)
        condition_number = lmbda2cn(lmbda)
        _, eigvals, A = compute_empirical_A_mc(pde, bcs, precond, lmbda, fnet, params, buffers, data_fd, fd = True)
        lr = 1 / eigvals[-1].item()
        print('Condition number', condition_number, 'lamopt', lmbda, 'lr', lr)
    else:
        lmbda = float(args.lmbda)
        condition_number = None
        # _, eigvals, A = compute_empirical_A(pde, bcs, precond, lmbda, fnet, params, buffers, data)
        # plt.figure(figsize=(30, 30))
        # plt.imshow(A.cpu().detach())
        # plt.colorbar()
        # plt.title('A')
        # plt.show()
        # plt.title('A')
        # plt.hist(eigvals.cpu().detach(), bins=100, alpha=0.7, color='blue', edgecolor='black')
        # plt.show()
        lr = float(args.lr)

    print(f'Using lmbdas : {lmbda}')

    losses_fn = partial(losses, pde, bcs, lmbda)
    sum_losses_fn = lambda *args: sum(losses_fn(*args))
    # plot_hessian(sum_losses_fn, (fnet, params, buffers, data), argnums=1, title='hess')
    if args.precond_on == 'grads' and args.precond != 'none':
        precond_pde_loss = lambda params: sum_losses_fn(fnet, precond(params, 'half'), buffers, data)
        # plot_hessian(precond_pde_loss, (params,), argnums=0, title='precond hess')

    if args.optimizer == 'adam':
        optimizer = torchopt.adam(lr=lr)
    elif args.optimizer == 'sgd':
        optimizer = torchopt.sgd(lr=lr, momentum=0)
    else:
        raise NotImplementedError(args.optimizer)

    opt_state = optimizer.init(params)
    loss_data = {'loss_res': [], 'loss_bc': [], 'loss': [], 'epoch': []}

    loss_res, loss_bc = losses_fn(fnet, params, buffers, data)
    loss_data['loss_res'].append(loss_res.item())
    loss_data['loss_bc'].append(loss_bc.item())
    loss_data['loss'].append((loss_res + loss_bc).item())
    loss_data['epoch'].append(0)

    mod = args.nepochs//min(args.nepochs, 500)
    for epoch in range(1, args.nepochs+1):
        loss_res, loss_bc, params, opt_state = step(optimizer, opt_state, precond,
                                                    losses_fn, fnet, params, buffers, data)
                
        if epoch%mod == 0:
            loss_data['loss_res'].append(loss_res)
            loss_data['loss_bc'].append(loss_bc)
            loss_data['loss'].append(loss_res + loss_bc)
            loss_data['epoch'].append(epoch)

            print(f'Epoch {epoch}, loss_res: {loss_res}, loss_bc: {loss_bc/lmbda}')
        # if epoch % 1000 == 0:
        #     _, eigvals, A = compute_empirical_A_mc(pde, bcs, precond, lmbda, fnet, params, buffers, data)
        #     plt.plot(fnet(params, buffers, data.x_interior).detach().cpu())
        #     plt.show()

    plt.plot(fnet(params, buffers, data.x_interior).detach().cpu())
    plt.show()

    results = {
        'final_loss_res': float(loss_res),
        'final_loss_bc': float(loss_bc),
        'final_loss': float(loss_res + loss_bc),
        'file': os.path.basename(__file__),
        'lr': lr,
        'condition_number': condition_number,
    }
    save(args.output, args, results, loss_data)
    torch.save(params, os.path.join(args.output, 'params'))
    torch.save(buffers, os.path.join(args.output, 'buffers'))