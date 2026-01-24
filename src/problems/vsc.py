from pathlib import Path

import numpy as np
import torch
from torch.distributions import Normal

from src.problems.problem import Problem


class VSC(Problem):
    def __init__(self, usecols=None) -> None:
        super().__init__()

        if usecols is None:
            usecols = [1, 2, 3, 4, 5]
        self.usecols = usecols

        current_file = Path(__file__).resolve()
        x_data, y_data = self._load_data(current_file.parent)

        self.x_data = x_data
        self.y_data = y_data

        self.ndim = None  # update in target

    def _load_data(self, folder):
        all_data = np.loadtxt(str(folder / "vs" / "six_dim_rr.csv"), delimiter=",", skiprows=1, usecols=self.usecols)
        x_data = all_data[:, 1:]
        y_data = all_data[:, 0]
        return x_data, y_data

    def target(self, **kwargs):
        target_ = TargetConditionalRobustBlockVSModel(self.y_data, self.x_data, **kwargs)

        self.ndim = target_.ndim

        return target_


class TargetConditionalRobustBlockVSModel:
    def __init__(self, y_data, x_data, device="cpu"):
        """
        Initialize Variables Selection target distribution with flexible model states.

        Parameters
        ----------
        y_data : random response variable, (n_observations,)
        x_data : design matrix, shape (n_observations, n_predictors)
        """
        self.device = device
        self.dtype: torch.dtype | None = None

        # Convert input data to tensor format
        self.y_data = self._to_tensor(y_data)
        self.x_data = self._to_tensor(x_data)
        assert self.x_data.shape[1] == 4, "x_data must have 4 columns (beta00,beta10,beta20,beta21)"

        self.ndim = 4  # Dimension of parameter space

        # Reference distribution
        self.sigma_prior = 10.0

    def log_prob(self, beta, context):
        """
        Compute log p(beta | D, model=context)

        Parameters
        ----------
        beta : torch.Tensor, shape (n_samples, 4)
            Full parameter vector (even inactive dims have values, but should be ~prior)
        context : torch.Tensor, (n_samples, 3) float mask

        Returns
        -------
        log_p : torch.Tensor, shape (n_samples,)
        """
        device = beta.device
        dtype = beta.dtype

        # --- Build prior on correct device ---
        loc = torch.tensor(0.0, device=device, dtype=dtype)
        scale = torch.tensor(self.sigma_prior, device=device, dtype=dtype)
        prior_dist = Normal(loc, scale)

        # --- Get mask: (n_samples, 4) ---
        if context.shape[1] != 3:
            raise ValueError("context must be (n,3) mask")

        repeats = torch.tensor([1, 1, 2], device=context.device)
        gamma = context.repeat_interleave(repeats, dim=1)  # (n_samples, 4)

        # --- Prior: all dims ~ N(0, sigma_prior^2) ---
        # Note: In true Bayesian VS, inactive params often have same prior
        # The log prior is the sum of the log prior densities for all parameters (active and inactive).
        log_prior = prior_dist.log_prob(beta).sum(dim=-1)  # (n_samples,)

        # --- Likelihood: only active params matter ---
        # Zero out inactive coefficients
        beta_active = beta * gamma  # (n_samples, 4)

        # Ensure self.x_data, self.y_data on correct device & dtype!
        x_data = self.x_data.to(device=device, dtype=dtype)
        y_data = self.y_data.to(device=device, dtype=dtype)

        # Linear predictor: (n_obs, n_samples)
        eta = torch.matmul(x_data, beta_active.T)  # (n_obs, 4) x (4, n_samples) = (n_obs, n_samples)
        resid = y_data.unsqueeze(1) - eta  # (n_obs, 1) - (n_obs, n_samples) = (n_obs, n_samples)

        # Robust mixture likelihood: N(0,1) + 0.1 * N(0,100)
        # p(y|x, \beta) = 1/\sqrt{2\pi} \exp(-r^2/2) + 0.1 * 1/\sqrt{200\pi} \exp(-r^2/200)
        # But constants cancel in log_prob (up to additive const), so:
        term1 = torch.exp(-(resid**2) / 2.0)
        term2 = 0.1 * torch.exp(-(resid**2) / 200.0)
        likelihood = term1 + term2

        # Log-sum for numerical stability (optional but better)
        # Here we use log(mn + eps) since mixture is simple
        log_likelihood = torch.log(likelihood + 1e-12).sum(dim=0)  # (n_samples,)

        return log_prior + log_likelihood

    def _to_tensor(self, data):
        """Convert input data to torch.Tensor with gradient support."""
        if isinstance(data, torch.Tensor):
            return data
        elif hasattr(data, "values"):
            return torch.tensor(data.values, dtype=torch.float32)
        else:
            return torch.tensor(data, dtype=torch.float32)
