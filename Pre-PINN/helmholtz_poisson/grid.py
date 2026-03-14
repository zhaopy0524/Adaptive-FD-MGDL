import math
import torch
from itertools import combinations


def create_grid(name):
    if name == '1d':
        return Grid1d()


def _create_interior_grid(extent, resolution, device):
    dim_ext = len(extent.shape)
    dim_res = len(resolution.shape)
    assert dim_ext == dim_res, 'Dimension of the domain extent and the domain resolution must be the same.'

    extent = extent[None] if dim_ext == 0 else extent
    steps = resolution[None] + 2 if dim_res == 0 else resolution + 2
    axes = [torch.linspace(0, ext, step, device = device)[1:-1] for ext, step in zip(extent, steps)]
    axes = torch.meshgrid(*axes, indexing = 'ij')
    return torch.cat([ax.flatten().unsqueeze(-1) for ax in axes], dim = -1)


class Grid1d:
    def __init__(self, res, device):
        self.extent = (-math.pi, math.pi)
        self.x_boundary_left = torch.tensor(self.extent[0]).view(-1, 1).requires_grad_(True).to(device)
        self.x_boundary_right = torch.tensor(self.extent[1]).view(-1, 1).requires_grad_(True).to(device)
        self.x_interior = torch.linspace(self.extent[0], self.extent[1], res).view(-1, 1).requires_grad_(True).to(device)
        self.x_boundaries = (self.x_boundary_left, self.x_boundary_right)

class Grid:
    def __init__(self, extent, res, device):
        self.device = device
        self.extent = extent.squeeze()
        self.resolution = res.squeeze()

        self.x_interior, self.x_boundaries = self._create_grid()

    def _create_grid(self):
        # Create interior grid
        x_interior = _create_interior_grid(self.extent, self.resolution, self.device).requires_grad_(True)

        # Create boundary grids
        dim = len(self.extent.shape)

        if dim == 0:
            x_boundaries = [
                torch.tensor([0.], device = self.device, requires_grad = True),
                torch.tensor([self.extent], device = self.device, requires_grad = True)]
        else:
            extent = list(self.extent.numpy())
            resolution = list(self.resolution.numpy())
            comb_ext = [list(comb) for comb in list(combinations(extent, 3))]
            comb_res = [list(comb) for comb in list(combinations(resolution, 3))]
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