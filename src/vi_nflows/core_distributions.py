import numpy as np
import torch
import torch.nn as nn
from normflows.distributions import BaseDistribution


class DiagGaussian(BaseDistribution):
    """
    Multivariate Gaussian distribution with diagonal covariance matrix.

    Extends normflows BaseDistribution with trainable location and scale parameters.
    """

    def __init__(self, shape, initial_loc=None, trainable=True):
        """
        Initialize diagonal Gaussian distribution.

        Parameters
        ----------
        shape : int or tuple
            Shape of the distribution. If int, converted to 1D tuple.
        initial_loc : torch.Tensor or None, optional
            Initial location parameter. If None, initialized to zeros.
        trainable : bool, optional
            Whether parameters should be trainable
        """
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.n_dim = len(shape)
        self.d = np.prod(shape)
        if trainable:
            if initial_loc is None:
                self.loc = nn.Parameter(torch.zeros(1, *self.shape))
            else:
                assert initial_loc.shape[0] == 1 and initial_loc.shape[1] == shape[0]
                self.loc = nn.Parameter(initial_loc)
            self.log_scale = nn.Parameter(torch.zeros(1, *self.shape))
        else:
            assert initial_loc is not None
            self.register_buffer("loc", initial_loc)
            self.register_buffer("log_scale", torch.zeros(1, *self.shape))
        self.temperature = None  # Temperature parameter for annealed sampling

    def forward(self, num_samples=1, context=None):
        """
        Sample from the distribution and compute log probabilities.

        Parameters
        ----------
        num_samples : int, optional
            Number of samples to generate

        Returns
        -------
        tuple
            (samples, log_probabilities)
        """
        eps = torch.randn((num_samples,) + self.shape, dtype=self.loc.dtype, device=self.loc.device)
        if self.temperature is None:
            log_scale = self.log_scale
        else:
            log_scale = self.log_scale + np.log(self.temperature)
        z = self.loc + torch.exp(log_scale) * eps
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow(eps, 2), list(range(1, self.n_dim + 1))
        )
        return z, log_p

    def log_prob(self, z, context=None):
        """
        Compute log probability of given samples.

        Parameters
        ----------
        z : torch.Tensor
            Input samples

        Returns
        -------
        torch.Tensor
            Log probabilities of the samples
        """
        if self.temperature is None:
            log_scale = self.log_scale
        else:
            log_scale = self.log_scale + np.log(self.temperature)
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow((z - self.loc) / torch.exp(log_scale), 2),
            list(range(1, self.n_dim + 1)),
        )
        return log_p


class DiagStudentT(BaseDistribution):
    """
    Multivariate Student's t-distribution with diagonal covariance and trainable degrees of freedom.

    Each dimension can learn different degrees of freedom as suggested in:
    "Fat-Tailed Variational Inference with Anisotropic Tail Adaptive Flows", ICML, 2022
    """

    def __init__(self, shape, initial_loc=None, trainable=True):
        """
        Initialize diagonal Student's t-distribution.

        Parameters
        ----------
        shape : int or tuple
            Shape of the distribution. If int, converted to 1D tuple.
        initial_loc : torch.Tensor or None, optional
            Initial location parameter. If None, initialized to zeros.
        trainable : bool, optional
            Whether location and scale parameters should be trainable
        """
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.n_dim = len(shape)
        self.d = np.prod(shape)

        # Recommended initial value in "Tails of Lipschitz Triangular Flows", ICML, 2020
        initial_log_deg = torch.log(torch.ones(self.d) * (30.0 - 1.0))

        # Note that each dimension can learn a different degree of freedom as suggested in
        # "Fat-Tailed Variational Inference with Anisotropic Tail Adaptive Flows", ICML, 2022
        self.log_deg_freedom = nn.Parameter(initial_log_deg)
        initial_log_scales = torch.zeros(self.d)

        if trainable:
            if initial_loc is None:
                self.loc = nn.Parameter(torch.zeros(self.d))
            else:
                assert initial_loc.shape[0] == 1 and initial_loc.shape[1] == shape[0]
                self.loc = nn.Parameter(torch.squeeze(initial_loc))
            self.log_scale = nn.Parameter(initial_log_scales)
        else:
            assert initial_loc is not None
            self.register_buffer("loc", torch.squeeze(initial_loc))
            self.register_buffer("log_scale", initial_log_scales)

    def forward(self, num_samples=1, context=None):
        """
        Sample from the distribution and compute log probabilities.

        Parameters
        ----------
        num_samples : int, optional
            Number of samples to generate

        Returns
        -------
        tuple
            (samples, log_probabilities)
        """
        nu = 1.0 + torch.exp(self.log_deg_freedom)
        studentT = torch.distributions.studentT.StudentT(df=nu, loc=self.loc, scale=torch.exp(self.log_scale))

        z = studentT.rsample(torch.Size([num_samples]))
        log_p = studentT.log_prob(z)
        log_p = torch.sum(log_p, dim=1)

        return z, log_p

    def log_prob(self, z, context=None):
        """
        Compute log probability of given samples.

        Parameters
        ----------
        z : torch.Tensor
            Input samples

        Returns
        -------
        torch.Tensor
            Log probabilities of the samples
        """
        nu = 1.0 + torch.exp(self.log_deg_freedom)
        studentT = torch.distributions.studentT.StudentT(df=nu, loc=self.loc, scale=torch.exp(self.log_scale))

        nan_sample_ids_z = torch.isnan(torch.sum(z, dim=1))

        if torch.any(nan_sample_ids_z):
            z_tmp = torch.clone(z)
            z_tmp[nan_sample_ids_z, :] = -1.0
            log_p = studentT.log_prob(z_tmp)
            log_p = torch.sum(log_p, dim=1)
            log_p[nan_sample_ids_z] = torch.nan
        else:
            log_p = studentT.log_prob(z)
            log_p = torch.sum(log_p, dim=1)

        return log_p
