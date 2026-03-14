from torch.func import jacrev


class BoundaryConditions:
    def __init__(self, name):
        self.name = name

    def residuals_single(self, include_source, fnet_single, params, buffers, x):
        if self.name == 'mixed_left':
            x = x[0] # just take the left for 1d
            grad_x_fnet_single = jacrev(fnet_single, argnums=2)
            sources = (lambda x: 1, lambda x: 0)
            sources_x = [s(x) if include_source else 0 for s in sources]
            return ((fnet_single(params, buffers, x) - sources_x[0]),
                    grad_x_fnet_single(params, buffers, x).squeeze(-1) - sources_x[1])
        elif self.name == 'zero_bc':
            return (fnet_single(params, buffers, x[0]),
                    fnet_single(params, buffers, x[1]))
        raise NotImplementedError(self.name)

