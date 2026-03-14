import math
import torch
from itertools import combinations
from scipy.stats.qmc import Sobol


def create_grid(name):
    if name == '1d':
        return Grid1d()


def _create_interior_grid(extent, resolution, device):
    assert extent.shape == resolution.shape, 'Dimension of the domain extent and the domain resolution must be the same.'

    extent = extent.unsqueeze(0) if extent.ndim == 0 else extent
    steps = resolution.unsqueeze(0) + 2 if resolution.ndim == 0 else resolution + 2
    axes = [torch.linspace(0, ext, step, device = device)[1:-1] for ext, step in zip(extent, steps)]
    axes = torch.meshgrid(*axes, indexing = 'ij')
    return torch.cat([ax.flatten().unsqueeze(-1) for ax in axes], dim = -1)

class Grid:
    def __init__(self, extent, res, device):
        self.device = device
        self.extent = torch.tensor(extent).squeeze() if type(extent) != torch.Tensor else extent.squeeze()
        self.resolution = torch.tensor(res).squeeze() if type(res) != torch.Tensor else res.squeeze()

        self.x_interior, self.x_boundaries = self._create_grid()

    def _create_grid(self):
        # Create interior grid
        x_interior = _create_interior_grid(self.extent, self.resolution, self.device).requires_grad_(True)

        # Create boundary grids
        is_scalar = bool(self.extent.ndim)
        if is_scalar:
            dim = self.extent.shape[0]
        else:
            dim = 1

        if dim == 1:
            x_boundaries = [
                torch.tensor([0.], device = self.device, requires_grad = True),
                torch.tensor([self.extent], device = self.device, requires_grad = True)
            ]
        else:
            extent = list(self.extent.numpy())
            resolution = list(self.resolution.numpy())
            comb_ext = [list(comb) for comb in list(combinations(extent, dim - 1))]
            comb_res = [list(comb) for comb in list(combinations(resolution, dim - 1))]
            shifts = [[0.]*len(comb) for comb in comb_ext]
            x_boundaries = []
            for i in range(len(comb_ext)):
                comb_ext[-i-1].insert(i, 0.)
                comb_res[-i-1].insert(i, 1)
                shifts[-i-1].insert(i, extent[i])
                x_boundaries.insert(-i-1, _create_interior_grid(torch.tensor(comb_ext[-i-1]), torch.tensor(comb_res[-i-1]), self.device))
            shifts = [torch.tensor(shift, device = self.device) for shift in shifts] 
            x_boundaries += [boundary.clone() + shift for boundary, shift in zip(x_boundaries, shifts)]
            x_boundaries = [boundary.requires_grad_(True) for boundary in x_boundaries]

        return x_interior, x_boundaries
    
def _create_interior_sampling(extent, log2_res, device, seed):
    assert extent.shape == log2_res.shape, 'Dimension of the domain extent and the domain resolution must be the same.'

    extent = extent.unsqueeze(0) if extent.ndim == 0 else extent
    sampler = Sobol(extent.shape[0], seed = seed)
    sample = sampler.random_base2(m = torch.sum(log2_res).cpu().item())
    sample = torch.tensor(sample, dtype = torch.float)*extent.reshape(1, -1)
    return sample.to(device)
    
class Sobolev_sampling:
    def __init__(self, extent, log2_res, device, seed):
        self.device = device
        self.seed = seed
        self.extent = torch.tensor(extent).squeeze() if type(extent) != torch.Tensor else extent.squeeze()
        self.log2_resolution = torch.tensor(log2_res).squeeze() if type(log2_res) != torch.Tensor else log2_res.squeeze()

        self.x_interior, self.x_boundaries = self._create_grid()

    def _create_grid(self):
        # Create interior grid
        x_interior = _create_interior_sampling(self.extent, self.log2_resolution, self.device, self.seed).requires_grad_(False)
        # Create boundary grids
        is_scalar = bool(self.extent.ndim)
        if is_scalar:
            dim = self.extent.shape[0]
        else:
            dim = 1

        if dim == 1:
            x_boundaries = [
                torch.tensor([0.], device = self.device, requires_grad = True),
                torch.tensor([self.extent], device = self.device, requires_grad = True)
            ]
        else:
            extent = list(self.extent.numpy())
            resolution = list(self.log2_resolution.numpy())
            comb_ext = [list(comb) for comb in list(combinations(extent, dim - 1))]
            comb_res = [list(comb) for comb in list(combinations(resolution, dim - 1))]
            shifts = [[0.]*len(comb) for comb in comb_ext]
            x_boundaries = []
            for i in range(len(comb_ext)):
                comb_ext[-i-1].insert(i, 0.)
                comb_res[-i-1].insert(i, 1)
                shifts[-i-1].insert(i, extent[i])
                x_boundaries.insert(-i-1, _create_interior_sampling(torch.tensor(comb_ext[-i-1]), torch.tensor(comb_res[-i-1]), self.device, self.seed))
            shifts = [torch.tensor(shift, device = self.device) for shift in shifts] 
            x_boundaries += [boundary.clone() + shift for boundary, shift in zip(x_boundaries, shifts)]
            x_boundaries = [boundary.requires_grad_(False) for boundary in x_boundaries]

        return x_interior, x_boundaries
    
class Grid1d:
    def __init__(self, res, device):
        self.extent = (-math.pi, math.pi)
        self.x_boundary_left = torch.tensor(self.extent[0]).view(-1, 1).requires_grad_(True).to(device)
        self.x_boundary_right = torch.tensor(self.extent[1]).view(-1, 1).requires_grad_(True).to(device)
        self.x_interior = torch.linspace(self.extent[0], self.extent[1], res).view(-1, 1).requires_grad_(True).to(device)
        self.x_boundaries = (self.x_boundary_left, self.x_boundary_right)