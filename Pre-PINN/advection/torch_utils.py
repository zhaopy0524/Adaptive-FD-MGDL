import copy

import torch
from typing import Iterable
from torch.nn.utils.convert_parameters import _check_param_device


def vector_to_parameters(vec: torch.Tensor, parameters: Iterable[torch.Tensor]):
    r"""
    As opposed to pytorch version, it is not inplace

    Convert one vector to the parameters

    Args:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(vec)))
    # Flag for the device where the parameter is located
    param_device = None
    # new_parameters = copy.deepcopy(parameters)
    new_tensors = []

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        new_tensor = vec[pointer:pointer + num_param].view_as(param)
        new_tensors.append(new_tensor)

        # Increment the pointer
        pointer += num_param

    return tuple(new_tensors)