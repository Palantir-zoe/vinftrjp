import numpy as np
import torch
from torch import Tensor, nn


class Target(nn.Module):
    """
    Sample target distributions to test models.

    Notes
    -----
    This class provides base functionality for target distributions used in
    testing normalizing flow models.
    """

    def __init__(self, n_dims: int, prop_scale: Tensor, prop_shift: Tensor):
        """
        Parameters
        ----------
        n_dims : int
            The dimensionality of Target
        prop_scale : torch.Tensor
            Scale for the uniform proposal distribution
        prop_shift : torch.Tensor
            Shift for the uniform proposal distribution
        """
        super().__init__()
        self.n_dims = n_dims
        self.prop_scale = prop_scale
        self.prop_shift = prop_shift

        self.register_buffer("prop_scale", prop_scale)
        self.register_buffer("prop_shift", prop_shift)

    def log_prob(self, z):
        """
        Compute log probability of the distribution.

        Parameters
        ----------
        z : torch.Tensor
            Value or batch of latent variables

        Returns
        -------
        torch.Tensor
            Log probability of the distribution for z

        Raises
        ------
        NotImplementedError
            If the method is not implemented in subclass
        """
        raise NotImplementedError("The log probability is not implemented yet.")

    def rejection_sampling(self, num_steps=1):
        """
        Perform rejection sampling on image distribution.

        Parameters
        ----------
        num_steps : int, optional
            Number of rejection sampling steps to perform, by default 1

        Returns
        -------
        torch.Tensor
            Accepted samples from the distribution
        """
        eps = torch.rand((num_steps, self.n_dims), dtype=self.prop_scale.dtype, device=self.prop_scale.device)
        z_ = self.prop_scale * eps + self.prop_shift
        prob = torch.rand(num_steps, dtype=self.prop_scale.dtype, device=self.prop_scale.device)
        prob_ = torch.exp(self.log_prob(z_) - self.max_log_prob)
        accept = prob_ > prob
        z = z_[accept, :]
        return z

    def sample(self, num_samples=1):
        """
        Sample from image distribution through rejection sampling.

        Parameters
        ----------
        num_samples : int, optional
            Number of samples to draw, by default 1

        Returns
        -------
        torch.Tensor
            Samples from the distribution
        """
        z = torch.zeros((0, self.n_dims), dtype=self.prop_scale.dtype, device=self.prop_scale.device)
        while len(z) < num_samples:
            z_ = self.rejection_sampling(num_samples)
            ind = np.min([len(z_), num_samples - len(z)])
            z = torch.cat([z, z_[:ind, :]], 0)
        return z
