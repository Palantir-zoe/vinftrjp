import torch


def set_requires_grad(module, flag):
    """Set requires_grad flag for all parameters in a torch.nn.module.

    Parameters
    ----------
    module : torch.nn.Module
        PyTorch module whose parameters will be modified.
    flag : bool
        Value to set for requires_grad attribute.

    Notes
    -----
    This function recursively sets the requires_grad attribute for all
    parameters in the module and its submodules. When flag is False,
    parameters will not be updated during gradient descent.

    Examples
    --------
    >>> model = torch.nn.Linear(10, 5)
    >>> set_requires_grad(model, False)  # Freeze model parameters
    >>> set_requires_grad(model, True)   # Unfreeze model parameters
    """
    for param in module.parameters():
        param.requires_grad = flag


def move_to_device(obj, data_type=None, device=None):
    """
    Move tensor to specified device and data type.

    Parameters
    ----------
    obj : torch.Tensor
        Tensor to move
    data_type : str or None, optional
        Data type: "float" or "double". If None, uses "float"
    device : str or None, optional
        Device: "cuda" or "cpu". If None, auto-detects

    Returns
    -------
    torch.Tensor
        Tensor moved to specified device and data type
    """
    # Set default values
    if data_type is None:
        data_type = "float"  # Default to float

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if data_type == "double":
        obj = obj.double()
    else:
        assert data_type == "float"
        obj = obj.float()

    obj = obj.to(device=device)
    return obj
