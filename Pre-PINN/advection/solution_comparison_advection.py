import torch
from net import FourierBasis, MLP
from utils import make_functional_with_buffers
from grid import Grid
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_advection_comparison(beta):
    xgrid, nt = 512, 1024

    device = 'cuda:1'
    extent = torch.tensor([2*torch.pi, 1.])
    nfreqs = torch.tensor([5, 30])
    resolution = torch.tensor([xgrid, nt])
    grid = Grid(extent, resolution, device)

    fourier = FourierBasis(extent, nfreqs).to(device)
    root = 'results/condn_vs_betas/advection/advection_2d_params/5_30_256_100/betas_' + str(beta) + '/'
    fnet, params, buffers = make_functional_with_buffers(fourier)
    params, buffers = torch.load(root + 'params'), torch.load(root + 'buffers')
    fb_precond = fnet(params, buffers, grid.x_interior).reshape(*resolution).detach().cpu().numpy().T

    device = 'cuda:0'
    extent = torch.tensor([2*torch.pi, 1.])
    nfreqs = torch.tensor([5, 30])
    resolution = torch.tensor([xgrid, nt])
    grid = Grid(extent, resolution, device)

    fourier = FourierBasis(extent, nfreqs).to(device)
    root = 'results/condn_vs_betas/advection/none_params/5_30_256_100/betas_' + str(beta) + '/'
    fnet, params, buffers = make_functional_with_buffers(fourier)
    params, buffers = torch.load(root + 'params'), torch.load(root + 'buffers')
    fb_nocond = fnet(params, buffers, grid.x_interior).reshape(*resolution).detach().cpu().numpy()

    device = 'cuda:1'
    extent = torch.tensor([2*torch.pi, 1.])
    resolution = torch.tensor([512, 1024])
    grid = Grid(extent, resolution, device)

    mlp = MLP(2, 1).to(device)
    root = 'results/condn_vs_betas/advection/none_params/mlp_256_100/betas_' + str(beta) + '/'
    fnet, params, buffers = make_functional_with_buffers(mlp)
    params, buffers = torch.load(root + 'params'), torch.load(root + 'buffers')
    mlp_nocond = fnet(params, buffers, grid.x_interior).reshape(*resolution).detach().cpu().numpy()

    extent = torch.tensor([2*torch.pi, 1.])
    resolution = torch.tensor([xgrid, nt])
    grid = Grid(extent, resolution, 'cpu')
    u_exact = torch.sin(grid.x_interior[:, 0] - beta*2*torch.pi*grid.x_interior[:, -1]).reshape(*resolution).detach().numpy()

    N = xgrid
    h = 2 * torch.pi / N
    x = torch.arange(0, 2*torch.pi, h) # not inclusive of the last point
    t = torch.linspace(0, 1, nt)

    fig, ax = plt.subplots(2, 2, figsize=(2*11, 2*7))


    h = ax[0, 0].imshow(u_exact, interpolation='nearest', cmap='viridis',
                    extent=[t.min(), t.max(), x.min(), x.max()],
                    origin='lower', aspect='auto')
    h.set_clim(-1, 1)
    divider = make_axes_locatable(ax[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)


    line = torch.linspace(x.min(), x.max(), 2)[:,None]

    ax[0, 0].set_xlabel('Time', size = 15)
    ax[0, 0].set_ylabel('Space', size = 15)
    ax[0, 0].tick_params(labelsize=15)
    ax[0, 0].set_title(r'Exact solution for $\beta = $' + str(beta), size = 20)

    h = ax[0, 1].imshow(fb_precond.T, interpolation='nearest', cmap='viridis',
                    extent=[t.min(), t.max(), x.min(), x.max()],
                    origin='lower', aspect='auto')
    h.set_clim(-1, 1)
    divider = make_axes_locatable(ax[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    line = torch.linspace(x.min(), x.max(), 2)[:,None]

    ax[0, 1].set_xlabel('Time', size = 15)
    ax[0, 1].set_ylabel('Space', size = 15)
    ax[0, 1].tick_params(labelsize=15)
    ax[0, 1].set_title('Fourier features with parameter preconditioning', size = 20)

    h = ax[1, 0].imshow(fb_nocond, interpolation='nearest', cmap='viridis',
                    extent=[t.min(), t.max(), x.min(), x.max()],
                    origin='lower', aspect='auto')
    h.set_clim(-1, 1)
    divider = make_axes_locatable(ax[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    line = torch.linspace(x.min(), x.max(), 2)[:,None]

    ax[1, 0].set_xlabel('Time', size = 15)
    ax[1, 0].set_ylabel('Space', size = 15)
    ax[1, 0].tick_params(labelsize=15)
    ax[1, 0].set_title('Fourier features without preconditioning', size = 20)

    h = ax[1, 1].imshow(mlp_nocond, interpolation='nearest', cmap='viridis',
                    extent=[t.min(), t.max(), x.min(), x.max()],
                    origin='lower', aspect='auto')
    h.set_clim(-1, 1)
    divider = make_axes_locatable(ax[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    line = torch.linspace(x.min(), x.max(), 2)[:,None]

    ax[1, 1].set_xlabel('Time', size = 15)
    ax[1, 1].set_ylabel('Space', size = 15)
    ax[1, 1].tick_params(labelsize=15)
    ax[1, 1].set_title('MLP without preconditioning', size = 20);

    fig.savefig('solution_comparison_advection_' + str(beta), format='pdf', bbox_inches='tight')

if __name__ == '__main__':
    for beta in [3, 15, 30]:
        plot_advection_comparison(beta)