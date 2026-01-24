import numpy as np
import torch
from normflows.flows import Flow

MIN_NR_MC_SAMPLES = 0


class TrainableLOFTLayer(Flow):
    """
    Trainable Linear Offset Transform (LOFT) layer.

    Implements a learnable transformation with softplus parameterization for
    improved numerical stability during training.
    """

    def __init__(self, dim, initial_t, train_t):
        """
        Initialize LOFT layer.

        Parameters
        ----------
        dim : int
            Dimensionality of the input data
        initial_t : float
            Initial value for the transformation parameter (must be >= 1.0)
        train_t : bool
            Whether the transformation parameter should be trainable
        """
        assert initial_t >= 1.0
        super().__init__()
        self.dim = dim
        self.rep_t = torch.ones(dim) * (initial_t - 1.0)  # reparameterization of t
        self.rep_t = torch.nn.Parameter(self.rep_t, requires_grad=train_t)
        return

    def forward(self, z):
        """
        Apply forward transformation.

        Parameters
        ----------
        z : torch.Tensor
            Input tensor of shape [batch_size, dim]

        Returns
        -------
        tuple
            (transformed_tensor, log_determinant) where:
            - transformed_tensor: Output after transformation
            - log_determinant: Log determinant of Jacobian
        """
        assert z.shape[0] >= MIN_NR_MC_SAMPLES  # batch size
        assert z.shape[1] >= 2  # theta should be at least of dimension 2

        t = self.get_t()

        new_value, part1 = self.loft_forward_static(t, z)

        log_derivatives = -torch.log(part1 + 1.0)

        log_det = torch.sum(log_derivatives, dim=1)

        return new_value, log_det

    def get_t(self):
        """
        Get transformation parameter with softplus constraint.

        Returns
        -------
        torch.Tensor
            Transformation parameter t = 1.0 + softplus(rep_t)
        """
        return 1.0 + torch.nn.functional.softplus(self.rep_t)

    @staticmethod
    def loft_forward_static(t, z):
        """
        Static method for LOFT forward transformation.

        Parameters
        ----------
        t : torch.Tensor
            Transformation parameter
        z : torch.Tensor
            Input tensor

        Returns
        -------
        tuple
            (transformed_value, part1) where:
            - transformed_value: Transformed output
            - part1: Intermediate computation used for log determinant
        """
        # $x = \text{sign}(z) \cdot (\log(\max(|z|-t,0)+1) + \min(|z|,t))$
        # log determinant: $\log|J| = -\sum \log(\max(|z|-t,0)+1)$
        part1 = torch.max(torch.abs(z) - t, torch.tensor(0.0))
        part2 = torch.min(torch.abs(z), t)

        new_value = torch.sign(z) * (torch.log(part1 + 1) + part2)

        return new_value, part1

    def inverse(self, z):
        """
        Apply inverse transformation.

        Parameters
        ----------
        z : torch.Tensor
            Input tensor of shape [batch_size, dim]

        Returns
        -------
        tuple
            (inverse_transformed_tensor, log_determinant) where:
            - inverse_transformed_tensor: Output after inverse transformation
            - log_determinant: Log determinant of inverse Jacobian
        """
        assert z.shape[0] >= MIN_NR_MC_SAMPLES  # Monte Carlo Samples
        assert z.shape[1] >= 2  # theta should be at least of dimension 2

        t = self.get_t()

        part1 = torch.max(torch.abs(z) - t, torch.tensor(0.0))
        part2 = torch.min(torch.abs(z), t)

        new_value = torch.sign(z) * (torch.exp(part1) - 1.0 + part2)

        log_det = torch.sum(part1, dim=1)

        return new_value, log_det


class PositiveConstraintLayer(Flow):
    """
    Positive constraint layer for enforcing positivity on specific dimensions.

    Applies softplus transformation to selected dimensions to ensure positive values
    while leaving other dimensions unchanged.
    """

    def __init__(self, pos_constraint_ids, total_dim):
        """
        Initialize positive constraint layer.

        Parameters
        ----------
        pos_contraint_ids : torch.Tensor
            Indices of dimensions that should be constrained to positive values
        total_dim : int
            Total dimensionality of the input data
        """
        super().__init__()
        assert torch.is_tensor(pos_constraint_ids)

        self.total_dim = total_dim
        self.pos_constraint_ids = pos_constraint_ids
        self.no_constraint_ids = torch.tensor(np.delete(np.arange(total_dim), pos_constraint_ids))

    def forward(self, z):
        """
        Apply forward transformation with positive constraints.

        Parameters
        ----------
        z : torch.Tensor
            Input tensor of shape [batch_size, total_dim]

        Returns
        -------
        tuple
            (constrained_tensor, log_determinant) where:
            - constrained_tensor: Output with positive constraints applied
            - log_determinant: Log determinant of Jacobian
        """
        # $x = \log(1+e^z)$
        # derivative: $\sigma(z)$
        # log derivative: $z - \text{softplus}(z)$
        assert z.shape[0] >= MIN_NR_MC_SAMPLES  # Monte Carlo Samples
        assert z.shape[1] == self.total_dim  # dimension

        new_values_pos, log_det_each_dim = self.forward_one_dim(z[:, self.pos_constraint_ids])

        assert new_values_pos.shape[0] >= MIN_NR_MC_SAMPLES
        assert new_values_pos.shape[1] == self.pos_constraint_ids.shape[0]

        all_new_values = torch.zeros_like(z)
        all_new_values[:, self.no_constraint_ids] = z[:, self.no_constraint_ids]
        all_new_values[:, self.pos_constraint_ids] = new_values_pos

        assert log_det_each_dim.shape[0] >= MIN_NR_MC_SAMPLES
        assert log_det_each_dim.shape[1] == self.pos_constraint_ids.shape[0]

        log_det = torch.sum(log_det_each_dim, dim=1)

        assert log_det.shape[0] >= MIN_NR_MC_SAMPLES

        return all_new_values, log_det

    def inverse(self, z):
        """
        Apply inverse transformation to remove positive constraints.

        Parameters
        ----------
        z : torch.Tensor
            Input tensor of shape [batch_size, total_dim]

        Returns
        -------
        tuple
            (unconstrained_tensor, log_determinant) where:
            - unconstrained_tensor: Output without positive constraints
            - log_determinant: Log determinant of inverse Jacobian
        """
        # $z = \log(e^x - 1)$
        assert z.shape[0] >= MIN_NR_MC_SAMPLES  # Monte Carlo Samples
        assert z.shape[1] == self.total_dim  # dimension

        assert torch.all(z[:, self.pos_constraint_ids] >= 0.0)

        new_value_posOrNeg, log_det_each_dim = self.inverse_one_dim(z[:, self.pos_constraint_ids])

        assert new_value_posOrNeg.shape[0] >= MIN_NR_MC_SAMPLES
        assert new_value_posOrNeg.shape[1] == self.pos_constraint_ids.shape[0]

        all_new_values = torch.zeros_like(z)

        all_new_values[:, self.no_constraint_ids] = z[:, self.no_constraint_ids]
        all_new_values[:, self.pos_constraint_ids] = new_value_posOrNeg

        assert log_det_each_dim.shape[0] >= MIN_NR_MC_SAMPLES
        assert log_det_each_dim.shape[1] == self.pos_constraint_ids.shape[0]

        log_det = torch.sum(log_det_each_dim, dim=1)

        assert log_det.shape[0] >= MIN_NR_MC_SAMPLES

        return all_new_values, log_det

    def forward_one_dim(self, z):
        """
        Apply softplus transformation to enforce positivity.

        Parameters
        ----------
        z : torch.Tensor
            Input tensor for constrained dimensions

        Returns
        -------
        tuple
            (positive_values, log_derivative) where:
            - positive_values: Transformed positive values
            - log_derivative: Log derivative for determinant calculation
        """
        new_value = torch.nn.functional.softplus(z)
        log_derivative = z - new_value

        return new_value, log_derivative

    def inverse_one_dim(self, x):
        """
        Apply inverse softplus transformation.

        Parameters
        ----------
        x : torch.Tensor
            Positive input tensor

        Returns
        -------
        tuple
            (unconstrained_values, log_derivative) where:
            - unconstrained_values: Transformed unconstrained values
            - log_derivative: Log derivative for determinant calculation
        """
        new_value = torch.log(torch.special.expm1(x))
        log_derivative = x - new_value

        return new_value, log_derivative


# Part of the code here is adapated from normflows package
# https://github.com/VincentStimper/normalizing-flows
class MaskedAffineFlowThresholded(Flow):
    """
    Masked affine flow with thresholding for improved numerical stability.

    Implements RealNVP-style affine transformations with various thresholding
    variations to prevent numerical instability during training.

    References
    ----------
    .. [1] "Density estimation using Real NVP" - arXiv:1605.08803
    .. [2] "Guided Image Generation with Conditional Invertible Neural Networks" - arXiv:1907.02392
    """

    def __init__(self, b, t=None, s=None, threshold=None, variation=None):
        """
        Initialize masked affine flow with thresholding.

        Parameters
        ----------
        b : torch.Tensor
            Binary mask tensor of same size as latent data point (0s and 1s)
        t : callable or None, optional
            Translation mapping function (neural network). If None, no translation applied.
        s : callable or None, optional
            Scale mapping function (neural network). If None, no scale applied.
        threshold : float
            Threshold value for scale parameter clipping (must be >= 0.05)
        variation : str
            Type of thresholding: "symmetric", "asymmetric", or "tanh"
        """
        super().__init__()
        self.b_cpu = b.view(1, *b.size())
        self.register_buffer("b", self.b_cpu)

        assert variation is not None
        assert threshold is not None
        assert threshold >= 0.05

        self.variation = variation
        self.threshold = threshold

        if s is None:
            self.s = lambda x: torch.zeros_like(x)
        else:
            self.add_module("s", s)

        if t is None:
            self.t = lambda x: torch.zeros_like(x)
        else:
            self.add_module("t", t)

        return

    def forward(self, z):
        """
        Apply forward affine transformation.

        Parameters
        ----------
        z : torch.Tensor
            Input tensor

        Returns
        -------
        tuple
            (transformed_tensor, log_determinant) where:
            - transformed_tensor: Output after affine transformation
            - log_determinant: Log determinant of Jacobian
        """
        z_masked = self.b * z
        scale = self.s(z_masked)
        nan = torch.tensor(np.nan, dtype=z.dtype, device=z.device)
        scale = torch.where(torch.isfinite(scale), scale, nan)
        trans = self.t(z_masked)
        trans = torch.where(torch.isfinite(trans), trans, nan)

        scale, trans = self.limit(scale, trans)

        z_ = z_masked + (1 - self.b) * (z * torch.exp(scale) + trans)
        log_det = torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))

        return z_, log_det

    def limit(self, scale, trans):
        """
        Apply thresholding to scale parameters for numerical stability.

        Parameters
        ----------
        scale : torch.Tensor
            Scale parameters
        trans : torch.Tensor
            Translation parameters

        Returns
        -------
        tuple
            (clamped_scale, trans) where scale is clamped according to variation type
        """
        if self.variation == "symmetric":
            # RealNVP variation as proposed in "Guided Image Generation with Conditional Invertible Neural Networks"
            # https://arxiv.org/abs/1907.02392
            scale = soft_clamp_asym(scale, neg_alpha=self.threshold, pos_alpha=self.threshold)
        elif self.variation == "asymmetric":
            # proposed clippling method
            scale = soft_clamp_asym(scale, neg_alpha=2.0, pos_alpha=self.threshold)
        elif self.variation == "tanh":
            # used by ATAF method
            scale = torch.tanh(scale)
        else:
            assert False

        return scale, trans

    def inverse(self, z):
        """
        Apply inverse affine transformation.

        Parameters
        ----------
        z : torch.Tensor
            Input tensor

        Returns
        -------
        tuple
            (inverse_transformed_tensor, log_determinant) where:
            - inverse_transformed_tensor: Output after inverse transformation
            - log_determinant: Log determinant of inverse Jacobian
        """
        z_masked = self.b * z
        scale = self.s(z_masked)
        nan = torch.tensor(np.nan, dtype=z.dtype, device=z.device)
        scale = torch.where(torch.isfinite(scale), scale, nan)
        trans = self.t(z_masked)
        trans = torch.where(torch.isfinite(trans), trans, nan)

        scale, trans = self.limit(scale, trans)

        z_ = z_masked + (1 - self.b) * (z - trans) * torch.exp(-scale)
        log_det = -torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))
        return z_, log_det


def soft_clamp_asym(value, neg_alpha, pos_alpha):
    """
    Apply asymmetric soft clamping using arctan function.

    Parameters
    ----------
    value : torch.Tensor
        Input values to clamp
    neg_alpha : float
        Clamping parameter for negative values
    pos_alpha : float
        Clamping parameter for positive values

    Returns
    -------
    torch.Tensor
        Soft-clamped values with asymmetric bounds
    """
    posValues = 0.5 * (torch.sign(value) + 1.0)
    negValues = 0.5 * (-torch.sign(value) + 1.0)

    posValues = posValues * (2.0 * pos_alpha / torch.pi) * torch.arctan(value / pos_alpha)
    negValues = negValues * (2.0 * neg_alpha / torch.pi) * torch.arctan(value / neg_alpha)
    return negValues + posValues
