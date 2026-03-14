import numpy as np
from matplotlib import pyplot as plt
import warnings
import matplotlib
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

import argparse
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F

from save_utils import save
from utils import compute_empirical_A, make_functional_with_buffers
from plot import plot_hessian
from precond import Preconditioner
from net import create_net, FourierBasis1d
from grid import Grid1d
from pde import PDE
from bcs import BoundaryConditions


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        nhidden = 5
        self.lin1 = nn.Linear(in_dim, 12, bias=True)
        self.lin2 = nn.Linear(12, nhidden, bias=True)
        self.lin3 = nn.Linear(nhidden, out_dim, bias=True)
        self.act_fn = torch.tanh

    def forward(self, x):
        h = self.lin1(x)
        h = self.act_fn(h)
        h = self.lin2(h)
        h = self.act_fn(h)
        h = self.lin3(h)
        return h


class DeepFourierBasis1d(FourierBasis1d):
    def __init__(self, extent, n_freq, init='standard', precond=None):
        super().__init__(extent, n_freq, init=init)
        nhidden = 5
        self.lin1 = nn.Linear(2 * n_freq + 1, nhidden, bias=False)
        self.last_lin = nn.Linear(nhidden, 1, bias=False)
        self.act_fn = F.gelu
        self.precond = precond

    def forward(self, x):
        assert x.shape[-1] == 1
        h = self.fourier_features(x)
        if self.precond is not None:
            h = self.precond(h, 'right')
        h = self.lin1(h)
        h = self.act_fn(h)
        h = self.last_lin(h)
        return h


def compute_spread(args):
    torch.manual_seed(0)
    data = Grid1d(args.res, args.device)
    pde = PDE(args.pde, args.pde_param)
    bcs = BoundaryConditions(args.bcs)

    if args.net == 'mlp':
        net = MLP(data.x_interior.shape[-1], 1)
    elif args.net == 'deep_fourier_basis_1d':
        net = DeepFourierBasis1d(data.extent, args.nfreqs, precond=Preconditioner(args, data))
    else:
        net = FourierBasis1d(data.extent, args.nfreqs)
    net = net.to(args.device)

    fnet_no_precond, params, buffers = make_functional_with_buffers(net)
    precond = Preconditioner(args, data).to(args.device)
    if args.precond_on == 'params':
        fnet = lambda params, *args: fnet_no_precond(precond(params, 'half'), *args)
    else:
        fnet = fnet_no_precond

    if args.lmbda == 'use_golden_search':
        print('Optimizing for lambdas!')
        assert args.lr is None, 'you cannot optimize and choose a lr at the same time'
        lmbda2cn = lambda lmbda: compute_empirical_A(pde, bcs, precond, lmbda, fnet, params, buffers, data)[0].item()
        positive_lam2cn = lambda x: float('inf') if x < 1e-6 else lmbda2cn(x)
        lmbda = scipy.optimize.golden(positive_lam2cn, maxiter=10)
        condition_number = lmbda2cn(lmbda)
        _, eigvals, A = compute_empirical_A(pde, bcs, precond, lmbda, fnet, params, buffers, data)
        lr = 1 / eigvals[-1].item()
        print('Condition number', condition_number, 'lamopt', lmbda, 'lr', lr)
    else:
        lmbda = float(args.lmbda)
        lr = float(args.lr)
        condition_number, eigvals, A = compute_empirical_A(pde, bcs, precond, lmbda, fnet, params, buffers, data)
    if args.net == 'mlp':
        label = 'MLP'
    elif args.net == 'deep_fourier_basis_1d':
        label = 'MLP + Preconditioned Fourier Features'
    else:
        label = 'Fourier Features + Preconditioning'
    print('label', label)
    plot_spread_of_A(eigvals, label)


def plot_spread_of_A(eigvals, label):
    eigvals = eigvals.cpu().detach().numpy()
    eigvals[eigvals < 1e-9] = 1e-9
    eigvals = eigvals / max(eigvals)

    # Plotting
    plt.yscale('log')
    plt.grid(True, which="major", ls="--", c='gray')  # Notice I added "both" for major and minor grid lines
    plt.hist(eigvals, edgecolor='black', bins=np.linspace(0, 1, 81), alpha=0.5, label=label, density=True)  # Added alpha and label



if __name__ == '__main__':
    pde = 'poisson'
    # pde = 'helmholtz'
    dfb_args = argparse.Namespace(pde=f'{pde}', bcs='zero_bc', device='cuda', res=1000, nepochs=1000,
                              net='deep_fourier_basis_1d', optimizer='sgd', lr='0.001',
                              precond=f'{pde}_deep_fourier_basis_1d', precond_on='net',
                              lmbda='1', output='results/testsss', nfreqs=12, pde_param=6)
    fb_args = argparse.Namespace(pde=f'{pde}', bcs='zero_bc', device='cuda', res=1000, nepochs=1000,
                              net='fourier_basis_1d', optimizer='sgd', lr=None,
                              precond=f'{pde}_fourier_basis_1d', precond_on='grads',
                              lmbda='use_golden_search', output='results/testsss', nfreqs=12, pde_param=6)
    mlp_args = argparse.Namespace(pde=f'{pde}', bcs='zero_bc', device='cuda', res=1000, nepochs=200, net='mlp', optimizer='sgd',
                       lr='0.001', precond='none', precond_on='net', lmbda='10', output='results/testsss', nfreqs=16,
                       pde_param=6)
    # Title
    # plt.title('Spread of eigenvalues of $\mathbb{A}$ for MLP')
    # plt.savefig('results/eigenvalue_spread_mlp.pdf', format='pdf', bbox_inches='tight')
    # plt.savefig('results/eigenvalue_spread_mlp_precond.pdf', format='pdf', bbox_inches='tight')

    # Show the plot
    plt.figure(figsize=(6.4, 3.5))
    plt.title('Distribution of eigenvalues of $\mathbb{A}$')

    compute_spread(fb_args)
    compute_spread(mlp_args)
    compute_spread(dfb_args)

    # Spines and axis formatting
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().set_ylim([.5, 1e2 * 2])  # setting the y-limit to match the tick range

    # Tick settings
    y_ticks = [10 ** k for k in range(0, 3, 1)]  # make sure these values suit your data
    plt.gca().set_yticks(y_ticks)

    # Get the handles and labels from your plots
    handles, labels = plt.gca().get_legend_handles_labels()
    # Reorder them manually
    order = [1, 0, 2]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
               frameon=False, loc='upper center', bbox_to_anchor=(0.5, .85))


    # save_title = f'{pde}_condn_vs_epochs_notitle'
    save_title = f'{pde}_eigen_distribution'
    save_path = f"results/{save_title}.pdf"
    print('saving as.. ', save_path)

    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    # plt.show()
