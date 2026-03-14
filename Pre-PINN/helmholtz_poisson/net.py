import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from precond import poisson_precond, helmholtz_precond, Preconditioner
from utils import create_freq_span


def create_net(args, data):
    if args.net == 'fourier_basis_1d':
        return FourierBasis1d(data.extent, args.nfreqs)
    elif args.net == 'deep_fourier_basis_1d':
        if args.precond_on == 'net':
            return DeepFourierBasis1d(data.extent, args.nfreqs, precond=Preconditioner(args, data))
        else:
            return DeepFourierBasis1d(data.extent, args.nfreqs)
    elif args.net == 'learnable_fourier_feats_1d':
        return LearnableFourierFeatures1d(data.extent, args.nfreqs)
    elif args.net == 'mlp':
        return MLP(data.x_interior.shape[-1], 1)
    raise NotImplementedError(args.net)

class FourierBasis(nn.Module):
    # The dimension of the basis is defined by the length of the "extent" tensor
    # Be careful if you feed an input of dimension one with a basis of higher dimension
    # than one as it won't trigger an error but will return a false result.

    def __init__(self, extent, n_freq, init = 'standard', orthonormal = True):
        super().__init__()
        self.extent = torch.tensor(extent) if type(extent)  == int else extent.squeeze()

        dim_freq = 0 if type(n_freq) == int else n_freq.ndim
        self.n_freq = n_freq if dim_freq > 1 else (n_freq*torch.ones_like(self.extent)).squeeze()
        self.orthonormal = orthonormal
        self.last_lin = nn.Linear(2*int(torch.prod(self.n_freq + 1)) - 1, 1, bias = False)
        # self.register_buffer('freqs', create_freq_span(n_freq, self.extent))

        if init == 'standard':
            pass
        elif init == 'zero':
            self.last_lin.weight.data.zero_()
        else:
            raise NotImplementedError(init)
        
    def forward(self, x):
        basis = self.fourier_basis(x)
        return self.last_lin(basis)
    
    def fourier_basis(self, x):
        n_freq = self.n_freq.unsqueeze(0) if self.n_freq.ndim == 0 else self.n_freq
        extent = self.extent.unsqueeze(0) if self.extent.ndim == 0 else self.extent
        harmonics = [2*torch.pi*torch.arange(0, n_freq[i] + 1, step = 1)/extent[i] for i in range(n_freq.shape[0])]
        harmonics = torch.cartesian_prod(*harmonics).reshape(-1, n_freq.shape[0])
        harmonics = torch.tensordot(harmonics, x, dims = ([1], [-1]))[1:].T
        basis = torch.cat([
            torch.ones(size = (harmonics.shape[0], 1)), 
            torch.cos(harmonics), 
            torch.sin(harmonics)
        ], dim = 1)

        if self.orthonormal:
            ext = torch.prod(extent)
            normalization = torch.cat([
                torch.sqrt(ext)*torch.ones(size = (1, 1)),
                torch.sqrt(ext/2)*torch.ones(size = (1, 2*int(torch.prod(n_freq + 1) - 1)))
            ], dim = 1)
            return basis/normalization
        else:
            return basis

class FourierBasis1d(nn.Module):
    def __init__(self, extent, n_freq, init='standard'):
        super().__init__()
        self.extent = extent
        self.n_freq = n_freq
        self.last_lin = nn.Linear(2 * n_freq + 1, 1, bias=False)
        self.register_buffer('freqs', create_freq_span(n_freq, self.extent))

        if init == 'standard':
            pass
        elif init == 'zero':
            self.last_lin.weight.data.zero_()
        else:
            raise NotImplementedError(init)

    def forward(self, x):
        assert x.shape[-1] == 1
        h = self.fourier_features(x)
        out = self.last_lin(h)
        return out

    def fourier_features(self, x):
        norm = math.sqrt(math.pi)
        return torch.cat([
            torch.cos(self.freqs.unsqueeze(0) * x) / norm,
            torch.sin(self.freqs.unsqueeze(0) * x) / norm,
            1 / math.sqrt(2)  / norm + 0 * x,
        ], -1)


class DeepFourierBasis1d(FourierBasis1d):
    def __init__(self, extent, n_freq, init='standard', precond=None):
        super().__init__(extent, n_freq, init=init)
        # nhidden = 2 * n_freq + 1
        nhidden = 32
        self.lin1 = nn.Linear(2 * n_freq + 1, nhidden, bias=False)
        # self.lin1.weight.data.normal_()
        self.lin2 = nn.Linear(nhidden, nhidden, bias=True)
        # self.lin3 = nn.Linear(nhidden, nhidden, bias=True)
        self.last_lin = nn.Linear(nhidden, 1, bias=False)
        # self.last_lin = nn.Linear(2 * n_freq + 1, 1, bias=False)
        # self.last_lin.weight.data.normal_()
        self.act_fn = F.gelu
        self.precond = precond
        # self.last_lin.weight.data.zero_()
        self.precond_method = 'poisson'
        if self.precond_method is not None:
            if self.precond_method == 'poisson':
                precond = poisson_precond(self.n_freq, self.extent)
            elif self.precond_method == 'helmholtz':
                precond = helmholtz_precond(self.n_freq, self.extent, self.pde_param)
            else:
                raise NotImplementedError(self.precond_method)
            self.register_buffer('precondm', precond)

    def forward(self, x):
        assert x.shape[-1] == 1
        h = self.fourier_features(x)
        if self.precond is not None:
            h = self.precond(h, 'right')
            # h = self.precondm.unsqueeze(0) * h
        # h = h / math.sqrt(h.shape[-1])
        h = self.lin1(h)
        # h = h / math.sqrt(h.shape[-1])
        h = self.act_fn(h)
        h = self.lin2(h)
        h = self.act_fn(h)
        # h = self.lin3(h)
        # h = self.act_fn(h)
        h = self.last_lin(h)
        # h = h / math.sqrt(h.shape[-1])
        return h

class LearnableFourierFeatures1d(nn.Module):
    def __init__(self, extent, n_freq):
        super().__init__()
        self.extent = extent
        self.n_freq = n_freq
        nhidden = 128
        self.lin1 = nn.Linear(1, nhidden, bias=False)
        self.lin2 = nn.Linear(nhidden, 1, bias=False)
        self.bias = nn.Parameter(torch.empty(nhidden,))
        L = 2 * math.pi / (self.extent[1] - self.extent[0])
        # torch.nn.init.normal_(self.lin1.weight, std=3)
        torch.nn.init.uniform_(self.bias, -math.pi, math.pi)
        torch.nn.init.uniform_(self.lin1.weight, -n_freq * L, n_freq * L)

    def forward(self, x):
        # x: (l, 1) -> hfx: (l, n_features, 1)
        assert x.shape[-1] == 1
        w = self.lin1.weight.squeeze(1).unsqueeze(0)
        bias = w * self.bias.unsqueeze(0)
        h = torch.sin(self.lin1(x) + bias)
        out = self.lin2(h)
        return out


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        nhidden = 32
        self.lin1 = nn.Linear(in_dim, nhidden, bias=True)
        self.lin2 = nn.Linear(nhidden, nhidden, bias=True)
        self.lin3 = nn.Linear(nhidden, out_dim, bias=True)
        self.act_fn = torch.tanh

    def forward(self, x):
        h = self.lin1(x)
        h = self.act_fn(h)
        h = self.lin2(h)
        h = self.act_fn(h)
        h = self.lin3(h)
        return h
