import torch
from torch.func import jacrev, jacfwd

def get_source(name, parameter):
    if name == 'helmholtz':
        return lambda x: 0
    elif name == 'poisson':
        k = parameter
        return lambda x : - k ** 2 * torch.sin(k * x)
    elif name == 'advection':
        return lambda x: torch.tensor(0.)
    else:
        raise NotImplementedError(name)

def finite_difference(u, data):
    dx = (data.extent[1] - data.extent[0]) / (u.shape[0] - 1)
    du_dx = (u[1:] - u[:-1]) / dx
    return du_dx

def trace(tensor):
    return torch.sum(torch.diagonal(tensor, dim1 = -2, dim2 = -1), dim = -1)

def pde_residual(name, parameter, source, model, params, buffers, x):
    if name == 'poisson':
        laplacian = jacfwd(jacrev(model, argnums = 2), argnums = 2)(params, buffers, x)
        laplacian = trace(laplacian)
        if source == None:
            return laplacian
        else:
            return laplacian + source(x)

    elif name == 'helmholtz':
        laplacian = jacfwd(jacrev(model, argnums = 2), argnums = 2)(params, buffers, x)
        laplacian = trace(laplacian)
        if source == None:
            return laplacian + parameter**2*model(params, buffers, x)
        else:
            return laplacian + parameter**2*model(params, buffers, x) - source(x)

    elif name == 'advection':
        jacobian = jacrev(model, argnums = 2)(params, buffers, x)
        if jacobian.ndim == 2:
            jacobian = jacobian.T
        if source == None:
            return jacobian[-1] + (2*torch.pi*parameter*jacobian[:-1]).sum(dim = 0)
        else:
            return jacobian[-1] + (2*torch.pi*parameter*jacobian[:-1]).sum(dim = 0) - source(x)

    elif name == 'heat':
        jacobian = jacrev(model, argnums = 2)
        laplacian = jacfwd(jacobian, argnums = 2)(params, buffers, x)
        jacobian = jacobian(params, buffers, x)
        if jacobian.ndim == 2:
            jacobian = jacobian.T
            laplacian = laplacian[:, :-1, :-1]
        else:
            laplacian = laplacian[:-1, :-1]
        laplacian = trace(laplacian)
        if source == None:
            return jacobian[-1] - parameter*laplacian
        else:
            return jacobian[-1] - parameter*laplacian - source(x)

    elif name == 'wave':
        hessian = jacfwd(jacrev(model, argnums = 2), argnums = 2)(params, buffers, x)
        hessian = torch.diagonal(hessian, dim1 = -2, dim2 = -1)
        if hessian.ndim == 2:
            hessian = hessian.T
        if source == None:
            return hessian[-1] - parameter**2*hessian[:-1].sum(dim = 0)
        else:
            return hessian[-1] - parameter**2*hessian[:-1].sum(dim = 0) - source(x)

class PDE:
    def __init__(self, name, parameter):
        self.name = name
        self.parameter = parameter

    def residual_single(self, add_source, fnet_single, params, buffers, x):
        if add_source:
            return pde_residual(self.name, self.parameter,
                                get_source(self.name, self.parameter),
                                fnet_single, params, buffers, x)
        else:
            return pde_residual(self.name, self.parameter,
                                None, fnet_single, params, buffers, x)