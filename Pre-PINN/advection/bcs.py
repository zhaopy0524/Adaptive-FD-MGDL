import torch
from torch.func import jacrev, grad, vmap


def dirichlet_condition(frontier_func, fnet_single, params, buffers, x_boundary):
    return fnet_single(params, buffers, x_boundary) - frontier_func(x_boundary).squeeze()

batch_dirichlet = vmap(dirichlet_condition, in_dims = (None, None, None, None, 0), out_dims = 0)

def neumann_condition(normal, frontier_func, fnet_single, params, buffers, x_boundary):
    dfnet = grad(fnet_single, argnums = 2)(params, buffers, x_boundary)
    return torch.dot(dfnet, normal) - frontier_func(x_boundary).squeeze()

batch_neumann = vmap(neumann_condition, in_dims = (None, None, None, None, None, 0), out_dims = 0)

def periodic_condition(fnet_single, params, buffers, x_boundary_0, x_boundary_L):
    return fnet_single(params, buffers, x_boundary_0) - fnet_single(params, buffers, x_boundary_L)

batch_periodic = vmap(periodic_condition, in_dims = (None, None, None, 0, 0), out_dims = 0)

class BoundaryConditions:
    def __init__(self, name, extent):
        self.name = name
        self.extent = torch.tensor(extent).squeeze() if type(extent) != torch.Tensor else extent.squeeze()

    def residuals(self, include_source, fnet_single, params, buffers, x: list, norm = False):
        if self.name == 'mixed_left':
            x = x[0] # just take the left for 1d
            grad_x_fnet_single = jacrev(fnet_single, argnums=2)
            sources = (lambda x: 1, lambda x: 0)
            sources_x = [s(x) if include_source else 0 for s in sources]
            return ((fnet_single(params, buffers, x) - sources_x[0]).squeeze().unsqueeze(0),
                    (grad_x_fnet_single(params, buffers, x).squeeze(-1) - sources_x[1]).squeeze().unsqueeze(0))
        elif self.name == 'zero_bc':
            return (fnet_single(params, buffers, x[0]).squeeze().unsqueeze(0),
                    fnet_single(params, buffers, x[1]).squeeze().unsqueeze(0))
        elif self.name == 'advection_2d':
            x_init, norm_init = x[0], self.extent[0] / x[0].shape[0]

            x_boundary_0, x_boundary_L, norm_periodic = x[1], x[3], self.extent[1] / x[1].shape[0]
            if include_source:
                u0 = lambda x: torch.sin(x[0])
                res_init = batch_dirichlet(u0, fnet_single, params, buffers, x_init)
            else:
                zero = lambda x: torch.tensor(0)
                res_init = batch_dirichlet(zero, fnet_single, params, buffers, x_init)
            res_periodic = batch_periodic(fnet_single, params, buffers, x_boundary_0, x_boundary_L)

            if norm:
                return res_init * torch.sqrt(norm_init), res_periodic * torch.sqrt(norm_periodic)
            else:
                return res_init, res_periodic

        else:
            raise NotImplementedError(self.name)


