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


def create_freq_span(n_freq, extent, device='cpu'):
    L = 2 * math.pi / (extent[1] - extent[0])
    k = torch.linspace(1, n_freq, n_freq, device=device) * L
    return k


def compute_empirical_A(pde, bcs, precond, lmbda, fnet, params, buffers, data):

    def diff_phi(diff_op_single, fnet, params, buffers, x):

        def fnet_single(params, buffers, x):
            return fnet(params, buffers, x.unsqueeze(0)).squeeze(0)

        def diff_phi_single(fnet_single, params, buffers, x):
            jac_params_single = jacrev(fnet_single, argnums=0)
            jac_flat_fn = lambda *args: p2v(jac_params_single(*args))
            out = diff_op_single(jac_flat_fn, params, buffers, x)
            return out

        dphi = vmap(diff_phi_single, in_dims=(None, None, None, 0)) \
            (fnet_single, params, buffers, x)

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
    diff_op_bcs = partial(bcs.residuals_single, False)
    dphi_pde = diff_phi(diff_op_pde, fnet, params, buffers, data.x_interior)
    phis_bcs = diff_phi(diff_op_bcs, fnet, params, buffers, data.x_boundaries)
    dx = (data.extent[1] - data.extent[0]) / dphi_pde.shape[0]
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