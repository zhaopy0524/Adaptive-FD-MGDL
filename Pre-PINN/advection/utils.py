import copy
import math
import numpy as np
from functools import partial
import torch
from torch.func import jacrev, vmap
from torch.nn.utils import parameters_to_vector as p2v


def make_functional_with_buffers(mod, disable_autograd_tracking=False):
    params_dict = dict(mod.named_parameters())
    params_names = params_dict.keys()
    params_values = tuple(params_dict.values())

    buffers_dict = dict(mod.named_buffers())
    buffers_names = buffers_dict.keys()
    buffers_values = tuple(buffers_dict.values())

    stateless_mod = copy.deepcopy(mod)
    stateless_mod.to('meta')

    def fmodel(new_params_values, new_buffers_values, *args, **kwargs):
        new_params_dict = {name: value for name, value in zip(params_names, new_params_values)}
        new_buffers_dict = {name: value for name, value in zip(buffers_names, new_buffers_values)}
        return torch.func.functional_call(stateless_mod, (new_params_dict, new_buffers_dict), args, kwargs)

    if disable_autograd_tracking:
        params_values = torch.utils._pytree.tree_map(torch.Tensor.detach, params_values)
    return fmodel, params_values, buffers_values

def create_freq_span(n_freq, extent, device = 'cpu'):
    nfreq = torch.tensor(n_freq, device = device).squeeze() if type(n_freq) != torch.Tensor else n_freq.squeeze().to(device)
    ext = torch.tensor(extent, device = device).squeeze() if type(extent) != torch.Tensor else extent.squeeze().to(device)

    if ext.ndim == 0:
        harmonics = 2*torch.pi*torch.arange(1, nfreq + 1, step = 1)/ext
    elif ext.ndim == 1 and ext.shape[0] == 2:
        harmonics = [
            2*torch.pi*torch.arange(-nfreq[0], nfreq[0] + 1, step = 1)/ext[0],
            2*torch.pi*torch.arange(0, nfreq[1] + 1, step = 1)/ext[1]
        ]
        harmonics = torch.cartesian_prod(*harmonics).reshape(-1, nfreq.shape[0])
        mask = torch.ones_like(harmonics[:, 0], dtype = torch.bool)
        indices = torch.argwhere((harmonics[:, 0] <= 0)*(harmonics[:, 1] == 0))[:, 0]
        mask[indices] = False
        harmonics = harmonics[mask]
    else:
        harmonics = [2*torch.pi*torch.arange(-nfreq[i], nfreq[i] + 1, step = 1)/ext[i] for i in range(nfreq.shape[0])]
        harmonics = torch.cartesian_prod(*harmonics).reshape(-1, nfreq.shape[0])
        index = int(torch.prod(2*nfreq + 1)//2)
        harmonics = torch.cat([harmonics[:index], harmonics[index + 1:]], dim = 0)
    return harmonics.to(device)


def compute_empirical_A(pde, bcs, precond, lmbda, fnet, params, buffers, data):

    def diff_phi(diff_op_single, fnet, params, buffers, x):

        def fnet_single(params, buffers, x):
            return fnet(params, buffers, x.unsqueeze(0)).squeeze(0)

        def diff_phi_single(fnet_single, params, buffers, x):
            jac_params_single = jacrev(fnet_single, argnums=0)
            jac_flat_fn = lambda *args: p2v(jac_params_single(*args))
            out = diff_op_single(jac_flat_fn, params, buffers, x)
            return out

        batch_dphi = vmap(diff_phi_single, in_dims=(None, None, None, 0))
        dphi = batch_dphi(fnet, params, buffers, x)

        return dphi

    # # computing phis associated to the NTK, without the boudary term
    # phis = diff_phi(lambda fn, p, b, x: fn(p, b, x), fnet, params, buffers, data.x_interior)
    # nkt_xx = phis.matmul(phis.T)
    # ntk_eigvals, ntk_eigvects = torch.linalg.eigh(nkt_xx, UPLO='U')
    # import matplotlib.pyplot as plt
    # for i in range(10):
    #     eigvect = ntk_eigvects[ntk_eigvects.shape[0] - 1 - i].squeeze().detach().cpu()
    #     y = eigvect# * ntk_eigvals[ntk_eigvects.shape[0] - 1 - i].item()
    #     plt.plot(data.x_interior.squeeze().detach().cpu(), y)
    # plt.title('NTK eigs')
    # plt.show()
    # for i in np.linspace(0, nkt_xx.shape[0]-1, 20).astype(int):
    #     plt.scatter(data.x_interior[i].squeeze().detach().cpu(), [0])
    #     plt.plot(data.x_interior.squeeze().detach().cpu(), nkt_xx[:, i].detach().cpu())
    # plt.title('NTK values')
    # plt.show()
    # return None, None, None

    diff_op_pde = partial(pde.residual_single, False)
    diff_op_bcs = partial(bcs.residuals, False)
    dphi_pde = diff_phi(diff_op_pde, fnet, params, buffers, data.x_interior)
    phis_bcs = diff_phi(diff_op_bcs, fnet, params, buffers, data.x_boundaries)

    dx = data.extent/dphi_pde.shape[0]
    dphidphi_pde = dx * torch.einsum('bp, bq->pq', dphi_pde, dphi_pde)
    phiphi_bcs = sum([lmbda * torch.einsum('bp, bq->pq', phi, phi)
                      for phi in phis_bcs])

    A = 2 * (dphidphi_pde + phiphi_bcs)
    # import matplotlib.pyplot as plt
    # plt.imshow(A.detach().cpu())
    # plt.title('A')
    # plt.show()
    if precond.on == 'grads':
        BABT = precond(A, mode='interleave')
    else:
        BABT = A

    # plt.imshow(BABT.detach().cpu())
    # plt.title('BABT')
    # plt.show()

    eigenvals = torch.linalg.eigvalsh(BABT, UPLO='U')

    if eigenvals[0] < 0:
        if eigenvals[0] < -1e-6 * eigenvals[-1]:
            assert False, 'error, it should be positive'
        else:
            # assume its just numerical errors...
            eigenvals[0] = 0

    condition_number = eigenvals[-1] / eigenvals[0]

    return condition_number, eigenvals, BABT

def phi_bc(bcs, fnet_single, params, buffers, x: list):
    grad_fnet = lambda params, buffers, x: p2v(jacrev(fnet_single)(params, buffers, x))
    gnn_xs = bcs.residuals(False, grad_fnet, params, buffers, x, norm = True)
    return gnn_xs

def phi_pde(pde, fnet_single, params, buffers, x):
    grad_fnet = lambda params, buffers, x: p2v(jacrev(fnet_single)(params, buffers, x))
    dgnn_x = pde.residual_single(False, grad_fnet, params, buffers, x)
    return dgnn_x

def phi_pde_fd(pde, fnet_single, params, buffers, x, resolution):
    assert pde.name == 'advection'
    grad_fnet = lambda params, buffers, x: p2v(jacrev(fnet_single)(params, buffers, x))
    grad_fnet = vmap(grad_fnet, in_dims = (None, None, 0), out_dims = 0)

    out = grad_fnet(params, buffers, x)
    out = out.reshape(*resolution, out.shape[-1])
    dx, dt = x[0, :-1], x[0, -1]
    dphidt = (out[1:-1, 2:] - out[1:-1, :-2])/2/dt
    dphidx = (out[2:, 1:-1] - out[:-2, 1:-1])/2/dx
    dphi_pde = dphidt + 2*torch.pi*pde.parameter*dphidx
    return dphi_pde.reshape(-1, dphi_pde.shape[-1])

def phi_(fnet_single, params, buffers, x):
    grad_fnet = lambda params, buffers, x: p2v(jacrev(fnet_single)(params, buffers, x))
    return grad_fnet(params, buffers, x)

def compute_empirical_A_mc(pde, bcs, precond, lmbda, fnet, params, buffers, sampling, fd = False):
    def fnet_single(params, buffers, x):
        return fnet(params, buffers, x.unsqueeze(0)).squeeze(0)

    if fd:
        dphi_pde = phi_pde_fd(pde, fnet_single, params, buffers, sampling.x_interior, sampling.resolution).detach() # Needs sampling to be a grid
    else:
        batch_phi_pde = vmap(phi_pde, in_dims = (None, None, None, None, 0), out_dims = 0)
        dphi_pde = batch_phi_pde(pde, fnet_single, params, buffers, sampling.x_interior).detach()
    phis_bcs = phi_bc(bcs, fnet_single, params, buffers, sampling.x_boundaries)
    phis_bcs = [phi_bcs.detach() for phi_bcs in phis_bcs]

    volume = torch.prod(sampling.extent)
    dphidphi_pde = volume*torch.einsum('bp, bq -> pq', dphi_pde, dphi_pde)/dphi_pde.shape[0]
    phiphi_bcs = sum([lmbda*torch.einsum('bp, bq -> pq', phi, phi) for phi in phis_bcs])
    A = 2 * (dphidphi_pde + phiphi_bcs)
    if precond.on == 'grads':
        BABT = precond(A, mode='interleave')
    else:
        BABT = A
    eigenvals = torch.linalg.eigvalsh(BABT, UPLO='U')
    # eigenvals[eigenvals < 1e-6] = 0
    # condition_number = eigenvals[-1] / eigenvals[eigenvals != 0][0]
    condition_number = eigenvals[-1] / eigenvals[0]

    # batch_phi = vmap(phi_, in_dims = (None, None, None, 0), out_dims = 0)
    # phi = batch_phi(fnet_single, params, buffers, sampling.x_interior)
    # phiphi = volume*torch.einsum('bp, bq -> pq', phi, phi)/phi.shape[0]

    return condition_number, eigenvals, BABT
    # return condition_number, eigenvals, phiphi, BABT