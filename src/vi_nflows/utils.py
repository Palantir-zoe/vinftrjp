import os

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


def resolve_flow_training_device(device=None):
    """Resolve the device used for flow training.

    Precedence is:
    1. Explicit ``device`` argument
    2. ``FLOW_TRAIN_DEVICE`` environment variable
    3. ``"cpu"``

    The special value ``"auto"`` selects CUDA when available and otherwise CPU.
    Any unavailable CUDA request falls back to CPU so training scripts remain usable
    on machines without a GPU-enabled PyTorch build.
    """

    requested = device
    if requested is None:
        requested = os.environ.get("FLOW_TRAIN_DEVICE", "cpu")

    requested = str(requested).strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("Requested CUDA flow training, but CUDA is unavailable; falling back to CPU.")
        return "cpu"

    return requested or "cpu"


def resolve_flow_training_int(env_name, default, minimum=1):
    """Resolve an integer flow-training setting from the environment."""

    raw_value = os.environ.get(env_name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError:
        print(f"Ignoring invalid integer value for {env_name}: {raw_value!r}. Using default {default}.")
        return default

    if value < minimum:
        print(f"Ignoring {env_name}={value} because it is smaller than {minimum}. Using default {default}.")
        return default

    return value


def resolve_flow_training_bool(env_name, default):
    """Resolve a boolean flow-training setting from the environment."""

    raw_value = os.environ.get(env_name)
    if raw_value is None:
        return default

    value = str(raw_value).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False

    print(f"Ignoring invalid boolean value for {env_name}: {raw_value!r}. Using default {default}.")
    return default


def prepare_flow_for_inference(flow, device="cpu"):
    """Move a trained flow to an inference device before caching or reuse."""

    if hasattr(flow, "to"):
        flow = flow.to(device)
    elif device == "cpu" and hasattr(flow, "cpu"):
        flow = flow.cpu()

    if hasattr(flow, "eval"):
        flow.eval()

    return flow
