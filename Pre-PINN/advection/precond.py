from functools import partial

import torch
from torch.nn.utils import parameters_to_vector as p2v

from utils import create_freq_span
from torch_utils import vector_to_parameters as v2p


def apply_to_subset(fn, subset_indices, x):
    x_sub = tuple(x[i] for i in subset_indices)
    x_sub_precond = fn(x_sub)
    x_new = []
    cntr = 0
    for i, xi in enumerate(x):
        if i in subset_indices:
            x_new.append(x_sub_precond[cntr])
            cntr += 1
        else:
            x_new.append(x[i])
    return tuple(x_new)


def helmholtz_precond(nfreqs, extent, omega):
    assert omega == int(omega)
    freqs = create_freq_span(nfreqs, extent)
    mult_cos = 1 / torch.abs(freqs ** 2 - omega ** 2)
    mult_sin = 1 / torch.abs(freqs ** 2 - omega ** 2)
    mult_sin[omega == freqs] = 20 / omega
    mult_cos[omega == freqs] = 20
    mult_zero = torch.ones(1) * 1 / omega ** 2
    mult = torch.cat([mult_cos, mult_sin, mult_zero], 0)
    return mult

def advection2D_precond(nfreqs, extent, beta):
    freqs = create_freq_span(nfreqs, extent)
    mult_cos = torch.abs(freqs[:, 1] + 2*torch.pi*beta*freqs[:, 0])
    mult_cos[mult_cos < 1e-4] = 1
    mult_cos = 1/mult_cos
    mult_sin = mult_cos.clone()
    mult_zero = torch.ones(1)
    mult = torch.cat([mult_cos, mult_sin, mult_zero], 0)
    return mult

def poisson_precond(nfreqs, extent):
    freqs = create_freq_span(nfreqs, extent)
    mult_cos = 1 / torch.abs(freqs ** 2)
    mult_sin = 1 / torch.abs(freqs ** 2)
    mult_zero = torch.ones(1) * 20
    mult = torch.cat([mult_cos, mult_sin, mult_zero], 0)
    return mult


def create_P_matrix(args, data):
    if args.precond == 'helmholtz_fourier_basis_1d':
        mult = helmholtz_precond(args.nfreqs, data.extent, args.pde_param)
        P = torch.diag(mult)
    elif args.precond == 'poisson_fourier_basis_1d':
        # assert args.net == 'fourier_basis_1d', f'Instead we have {args.net}'
        mult = poisson_precond(args.nfreqs, data.extent)
        P = torch.diag(mult)
    elif args.precond == 'advection_2d':
        mult = advection2D_precond(args.nfreqs, data.extent, args.pde_param)
        P = torch.diag(mult)
    # elif name == 'poisson_deep_fourier_basis_1d':
    #     mult = poisson_precond(args.nfreqs, data.extent).to(args.device)
    #     P = torch.diag(mult)
    else:
        raise NotImplementedError(args.precond)
    return P


def P_mult(P, mode, x):
    if mode == 'half':
        return v2p(P.matmul(p2v(x)), x)
    elif mode == 'full':
        return v2p(P.matmul(P.T.matmul(p2v(x))), x)
    elif mode == 'interleave':
        # x is a matrix here... not super clean
        return P.matmul(x.matmul(P.T))
    elif mode == 'right':
        return x.matmul(P.T)
    else:
        raise NotImplementedError(mode)


class Preconditioner(torch.nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.name = args.precond
        self.on = args.precond_on
        if self.name != 'none':
            self.register_buffer('P', create_P_matrix(args, data))

    def forward(self, x, mode):
        if self.name != 'none':
            return P_mult(self.P, mode, x)
        else:
            return x