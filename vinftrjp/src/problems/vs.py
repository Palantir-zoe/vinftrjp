from pathlib import Path

import numpy as np
import torch
from torch.distributions import Normal

from src.problems.problem import Problem


class VS(Problem):
    def __init__(self, usecols=None) -> None:
        super().__init__()

        if usecols is None:
            usecols = [1, 2, 3, 4, 5]
        self.usecols = usecols

        current_file = Path(__file__).resolve()
        x_data, y_data = self._load_data(current_file.parent)

        self.x_data = x_data
        self.y_data = y_data

        self.ndim = None

    def _load_data(self, folder):
        all_data = np.loadtxt(str(folder / "vs" / "six_dim_rr.csv"), delimiter=",", skiprows=1, usecols=self.usecols)
        x_data = all_data[:, 1:]
        y_data = all_data[:, 0]
        return x_data, y_data

    def target(self, k: list[int], **kwargs):
        target_ = TargetRobustBlockVSModel(self.y_data, self.x_data, k=k, **kwargs)

        self.ndim = target_.ndim

        return target_


class TargetRobustBlockVSModel:
    def __init__(self, y_data, x_data, k=None):
        """
        Initialize Variables Selection target distribution with flexible model states.

        Parameters
        ----------
        y_data : random response variable, (n_observations,)
        x_data : design matrix, shape (n_observations, n_predictors)
        k : list of int, optional
            Model state indicator, e.g., [1,0,0] means beta0 is included,
            beta1 and beta2 are excluded. If None, uses default [1,1,1]
        """
        # Convert input data to tensor format
        self.y_data = self._to_tensor(y_data)
        self.x_data = self._to_tensor(x_data)

        # Model configuration
        self.nblocks = 3  # Three blocks: beta0, beta1, beta2
        self.blocksizes = [1, 1, 2]  # beta0: scalar, beta1: scalar, beta2: vector of length 2

        # Set model state based on k parameter
        if k is None:
            self.k = [1, 1, 1]  # Default: include all blocks
        else:
            if len(k) != self.nblocks:
                raise ValueError(f"k must have length {self.nblocks}, got {len(k)}")
            if not all(k_i in [0, 1] for k_i in k):
                raise ValueError("All elements of k must be 0 or 1")
            self.k = k

        # Calculate dimension based on active blocks
        self.active_blocksizes = [size for i, size in enumerate(self.blocksizes) if self.k[i] == 1]
        self.ndim = sum(self.active_blocksizes)  # Dimension of parameter space

        # Setup block names and parameter names
        self.blocknames = [f"b{i}" for i in range(self.nblocks)]
        self.active_blocknames = [f"b{i}" for i in range(self.nblocks) if self.k[i] == 1]

        # Generate parameter names based on active blocks
        self.betanames = []
        for i in range(self.nblocks):
            if self.k[i] == 1:  # Only include active blocks
                for j in range(self.blocksizes[i]):
                    self.betanames.append(f"beta{i}{j}")

    def get_active_parameters(self, theta):
        """
        Extract active parameters from theta based on model state.

        Parameters
        ----------
        theta : torch.Tensor, shape (n_samples, total_dim)

        Returns
        -------
        torch.Tensor, shape (n_samples, active_dim)
        """
        return theta  # In this implementation, theta already contains only active parameters

    def compute_prior(self, theta):
        """
        Compute log prior probability for active parameters.

        Parameters
        ----------
        theta : torch.Tensor, shape (n_samples, n_active_parameters)

        Returns
        -------
        torch.Tensor, shape (n_samples,)
            Log prior probability values.
        """
        # All active parameters have N(0, 10^2) prior
        if theta.shape[1] > 0:
            prior_log_prob = Normal(loc=0.0 * torch.ones_like(theta), scale=10.0 * torch.ones_like(theta)).log_prob(
                theta
            )
            prior_log_prob = torch.sum(prior_log_prob, dim=1)
        else:
            prior_log_prob = torch.zeros(theta.shape[0])

        return prior_log_prob

    def compute_llh(self, theta):
        """
        Compute log likelihood for the current model state.

        target <- function(x){
          p <- length(x)
          a <- X%*%x
          mn <- exp(-(y - a)^2/2) + exp(-(y - a)^2/200)/10
          phi_0 <- log(mn)
          log_q <- sum(phi_0) + sum(x^2/200)
          return(list(log_q = log_q))
        }
        """
        n_samples = theta.shape[0]
        n_data = self.y_data.shape[0]

        # Initialize log likelihood
        log_like = torch.zeros(n_samples)

        # For each sample, compute likelihood
        for i in range(n_samples):
            # Extract parameters for this sample
            sample_theta = theta[i : i + 1, :]  # Shape: (1, n_active_params)

            # Construct full beta vector based on model state
            beta_full = torch.zeros(4)  # beta0, beta1, beta20, beta21

            # Fill in active parameters
            param_idx = 0
            for block_idx in range(self.nblocks):
                if self.k[block_idx] == 1:
                    block_size = self.blocksizes[block_idx]
                    if block_idx == 0:  # beta0
                        beta_full[0] = sample_theta[0, param_idx]
                    elif block_idx == 1:  # beta1
                        beta_full[1] = sample_theta[0, param_idx]
                    elif block_idx == 2:  # beta2
                        beta_full[2] = sample_theta[0, param_idx]
                        beta_full[3] = sample_theta[0, param_idx + 1]
                    param_idx += block_size

            # Compute linear predictor
            a = torch.matmul(self.x_data, beta_full.unsqueeze(-1)).squeeze()

            # Compute mixture likelihood
            mn = torch.exp(-((self.y_data - a) ** 2) / 2) + torch.exp(-((self.y_data - a) ** 2) / 200) / 10
            log_mn = torch.log(mn)  # Add small constant for numerical stability

            # Sum over observations
            log_like[i] = torch.sum(log_mn)

        return log_like

    def log_prob(self, theta):
        """
        Compute total log probability (prior + likelihood).

        Parameters
        ----------
        theta : torch.Tensor, shape (n_samples, n_active_parameters)

        Returns
        -------
        torch.Tensor, shape (n_samples,)
            Total log probability values.
        """
        return self.compute_prior(theta) + self.compute_llh(theta)

    def _to_tensor(self, data):
        """Convert input data to torch.Tensor with gradient support."""
        if isinstance(data, torch.Tensor):
            return data
        elif hasattr(data, "values"):
            return torch.tensor(data.values, dtype=torch.float32)
        else:
            return torch.tensor(data, dtype=torch.float32)
