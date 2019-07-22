import torch

def _standard_normal(shape, dtype, device):
    if torch._C._get_tracing_state():
        # [JIT WORKAROUND] lack of support for .normal_()
        return torch.normal(torch.zeros(shape, dtype=dtype, device=device),
                            torch.ones(shape, dtype=dtype, device=device))
    return torch.empty(shape, dtype=dtype, device=device).normal_()


def _sum_rightmost(value, dim):
    r"""
    Sum out ``dim`` many rightmost dimensions of a given tensor.
    Args:
        value (Tensor): A tensor of ``.dim()`` at least ``dim``.
        dim (int): The number of rightmost dims to sum out.
    """
    if dim == 0:
        return value
    required_shape = value.shape[:-dim] + (-1,)
    return value.reshape(required_shape).sum(-1)
