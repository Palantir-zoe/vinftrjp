import numpy as np
import torch
from normflows.flows import Flow


class ConditionalMaskedAffineFlow(Flow):
    """
    Conditional masked affine flow layer.

    A conditional version of the MaskedAffineFlow that applies affine transformations
    only to dimensions specified by a binary mask, with transformation parameters
    conditioned on auxiliary information y (e.g., class labels, context vectors).

    The transformation is defined as:
        f(z|y) = b * z + (1 - b) * (z * exp(s(b * z, y)) + t(b * z, y))
        f(z|y) = b ⊙ z + (1 − b) ⊙ (z ⊙ exp(s(b ⊙ z, y)) + t(b ⊙ z, y))

    where:
        - b is the binary mask (1: preserve, 0: transform)
        - s(·, y) is the conditional scale function
        - t(·, y) is the conditional translation function

    This formulation enables different transformations for different conditions
    while maintaining exact invertibility.

    Parameters
    ----------
    b : torch.Tensor
        Binary mask tensor of shape (d,) where d is the feature dimension.
        Dimensions with value 0 will be transformed, dimensions with value 1
        will be preserved.
    s : torch.nn.Module or callable, optional
        Conditional scale function/mapping. Should accept concatenated
        [z_masked, y] as input and output scale parameters of the same shape
        as z. If None, no scaling is applied (s = 0, resulting in identity).
    t : torch.nn.Module or callable, optional
        Conditional translation function/mapping. Should accept concatenated
        [z_masked, y] as input and output translation parameters of the same
        shape as z. If None, no translation is applied (t = 0).

    Attributes
    ----------
    b : torch.Tensor
        Registered buffer for the binary mask of shape (1, d).
    s : callable
        Scale function, either a lambda returning zeros or a neural network module.
    t : callable
        Translation function, either a lambda returning zeros or a neural network module.
    """

    def __init__(self, b, t=None, s=None):
        """Initialize the conditional masked affine flow layer.

        Parameters
        ----------
        b : torch.Tensor
            Binary mask tensor of shape (d,) indicating which dimensions to transform.
        s : torch.nn.Module or callable, optional
            Conditional scale function. If None, scaling is disabled.
        t : torch.nn.Module or callable, optional
            Conditional translation function. If None, translation is disabled.
        """
        super().__init__()
        self.b_cpu = b.view(1, *b.size())
        self.register_buffer("b", self.b_cpu)

        if s is None:
            self.s = torch.zeros_like  # lambda x: torch.zeros_like(x)
        else:
            self.add_module("s", s)

        if t is None:
            self.t = torch.zeros_like  # lambda x: torch.zeros_like(x)
        else:
            self.add_module("t", t)

    def forward(self, z, context=None):
        """
        Apply forward transformation: z -> z' given condition y.

        Parameters
        ----------
        z : torch.Tensor
            Input tensor of shape [batch_size, d], where d is the feature dimension.
        context : torch.Tensor or None, optional
            Condition tensor. Can be of shape:
            - [batch_size, context_dim] (2D)
            If None, the transformation becomes unconditional.

        Returns
        -------
        z_ : torch.Tensor
            Transformed tensor of shape [batch_size, d].
        log_det : torch.Tensor
            Log determinant of the Jacobian of shape [batch_size].

        Notes
        -----
        The log determinant is computed as:
            log_det = sum_{i where b_i=0} s_i
        where s_i are the scale parameters for transformed dimensions.
        This follows from the diagonal structure of the Jacobian.
        """
        y_prepared = context

        # Apply mask: preserve dimensions where b = 1
        z_masked = self.b * z

        # Prepare input for conditional functions
        if y_prepared is not None:
            # Concatenate along feature dimension
            s_input = torch.cat([z_masked, y_prepared], dim=-1)
            t_input = torch.cat([z_masked, y_prepared], dim=-1)

            # Pass through modules
            scale = self.s(s_input)
            trans = self.t(t_input)

        else:
            # No condition: use unconditional transformation
            scale = self.s(z_masked)
            trans = self.t(z_masked)

        # Reshape back to original shape
        scale = scale.view_as(z)
        trans = trans.view_as(z)

        # Handle numerical stability: replace non-finite values with NaN
        nan = torch.tensor(np.nan, dtype=z.dtype, device=z.device)
        scale = torch.where(
            torch.isfinite(scale), scale, nan
        )  # This will cause NaNs to propagate, crashing training silently.
        trans = torch.where(
            torch.isfinite(trans), trans, nan
        )  # This will cause NaNs to propagate, crashing training silently.

        # Apply affine transformation to unmasked dimensions
        z_ = z_masked + (1 - self.b) * (z * torch.exp(scale) + trans)

        # Compute log determinant (sum of scales over transformed dimensions)
        log_det = torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))

        return z_, log_det

    def inverse(self, z, context=None):
        """
        Apply inverse transformation: z' -> z given condition y.

        Parameters
        ----------
        z : torch.Tensor
            Transformed tensor of shape [batch_size, d].
        context : torch.Tensor or None, optional
            Condition tensor (same format as in forward method).

        Returns
        -------
        z_ : torch.Tensor
            Original tensor of shape [batch_size, d].
        log_det : torch.Tensor
            Log determinant of the Jacobian of shape [batch_size].

        Notes
        -----
        The inverse transformation is exact and recovers the original input
        when using the same condition y. The log determinant for the inverse
        is the negative of the forward log determinant.
        """
        y_prepared = context

        # Apply mask
        z_masked = self.b * z

        # Prepare input for conditional functions (same as forward)
        if y_prepared is not None:
            s_input = torch.cat([z_masked, y_prepared], dim=-1)
            t_input = torch.cat([z_masked, y_prepared], dim=-1)

            scale = self.s(s_input)
            trans = self.t(t_input)

            scale = scale.view_as(z)
            trans = trans.view_as(z)

        else:
            scale = self.s(z_masked)
            trans = self.t(z_masked)

        # Handle numerical stability
        nan = torch.tensor(np.nan, dtype=z.dtype, device=z.device)
        scale = torch.where(torch.isfinite(scale), scale, nan)
        trans = torch.where(torch.isfinite(trans), trans, nan)

        # Apply inverse affine transformation
        z_ = z_masked + (1 - self.b) * (z - trans) * torch.exp(-scale)

        # Compute log determinant (negative of forward)
        log_det = -torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))

        return z_, log_det
