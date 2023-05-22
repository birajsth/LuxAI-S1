
" Some useful lib functions "

import numpy as np

import torch
from torch.autograd import Variable


def update_linear_schedule(optimizer, update, num_updates, initial_lr):
    """Decreases the learning rate linearly"""
    frac = 1.0 - (update - 1.0) / num_updates
    lrnow = frac * initial_lr
    optimizer.param_groups[0]["lr"] = lrnow
        
def np_one_hot(targets, nb_classes):
    """This is for numpy array
    https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy
    """

    print('nb_classes', nb_classes) if debug else None
    print('targets', targets) if debug else None

    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]

    return res.reshape(list(targets.shape) + [nb_classes])


def np_one_hot_fast(targets, nb_classes):
    """
    """

    print('nb_classes', nb_classes) if debug else None
    print('targets', targets) if debug else None

    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]

    return res.reshape(list(targets.shape) + [nb_classes])


def tensor_one_hot(labels, num_classes):
    """Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    cuda_check = labels.is_cuda
    if cuda_check:
        get_cuda_device = labels.get_device()

    y = torch.eye(num_classes)

    if cuda_check:
        y = y.to(get_cuda_device)

    return y[labels]


def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    print('y', y) if debug else None
    cuda_check = y.is_cuda
    print('cuda_check', cuda_check) if debug else None

    if cuda_check:
        get_cuda_device = y.get_device()
        print('get_cuda_device', get_cuda_device) if debug else None

    y_tensor = y.data if isinstance(y, Variable) else y
    print('y_tensor', y_tensor) if debug else None
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    print('y_tensor', y_tensor) if debug else None

    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)

    if cuda_check:
        y_one_hot = y_one_hot.to(get_cuda_device)

    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot



def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    # from https://github.com/allenai/allennlp/blob/b6cc9d39651273e8ec2a7e334908ffa9de5c2026/allennlp/nn/util.py
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result



def print_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}, Parameters: ", param.numel())
    total_params = sum( param.numel() for param in model.parameters())
    print(f"Total Parameters: {total_params}")



if __name__ == '__main__':
    pass