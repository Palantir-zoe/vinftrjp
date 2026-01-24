import torch
from torch.distributions import HalfNormal, InverseGamma, MultivariateNormal, Normal

from src.algorithms.fa_models import FactorAnalysisModel
from src.variables import ParametricModelSpace

from .problem import Problem


class FA(Problem):
    """
    Factor Analysis Problem Class

    Wrapper class for factor analysis problem setup and target distribution initialization.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize Factor Analysis problem.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for problem configuration
        """
        super().__init__()

        self.ndim = None

    def target(self, y_data, k, **kwargs):
        """
        Create target distribution for factor analysis.

        Parameters
        ----------
        y_data : array-like
            Observed data matrix
        k : int
            Number of latent factors

        Returns
        -------
        FactorAnalysisTarget
            Target distribution instance for factor analysis
        """
        target_ = FactorAnalysisTarget(y_data, k=k + 1)  # Compulsory

        self.ndim = target_.ndim

        return target_


class FactorAnalysisTarget(FactorAnalysisModel):
    """
    Bayesian Factor Analysis Target Distribution with gradient preservation.

    Implements the posterior distribution for factor analysis model with:
    - Lower triangular factor loading matrix
    - Diagonal noise covariance matrix
    - Hierarchical priors on parameters
    - Full gradient support for PyTorch optimization

    Attributes
    ----------
    y_data : torch.Tensor
        Observed data matrix of shape (N, d)
    k : int
        Number of latent factors
    obs_dim : int
        Observation dimension (d)
    ndim : int
        Total parameter dimension
    betaii_names : list
        Names of diagonal elements of factor loading matrix
    betaij_names : list
        Names of off-diagonal elements (lower triangular)
    lambda_names : list
        Names of diagonal noise variance elements
    blockrv : object
        Random variable block structure
    pos_constraint_ids : torch.Tensor
        Indices of parameters with positive constraints
    pos_param_list : list
        List of parameter names requiring positive constraints
    """

    def __init__(self, y_data, *, k):
        """
        Initialize Factor Analysis target distribution.

        Parameters
        ----------
        y_data : array-like, shape (N, d)
            Observed data matrix (assumed zero-mean for factor analysis).
        k : int
            Number of latent factors (must satisfy k <= d).

        Notes
        -----
        Total parameter dimension: d*(k+1) - k*(k-1)/2
        For d=6, k=2: 6*(2+1) - 2*1/2 = 18 - 1 = 17 parameters
        """
        # Convert input data to tensor format
        self.y_data = self._to_tensor(y_data)  # shape: (N, d)
        self.k = k
        self.obs_dim = self.y_data.shape[1]  # Observation dimension

        # Calculate total parameter dimension
        self.ndim = int(self.obs_dim * (k + 1) - k * (k - 1) / 2)

        # Parameter name lists for indexing
        self.betaii_names = []  # Diagonal elements of factor loading matrix
        self.betaij_names = []  # Off-diagonal elements (lower triangular)
        self.lambda_names = []  # Diagonal noise variance elements

        # Setup random variable structure and initialize parametric model space
        _, blockrv = self._setup_random_variables(k_min=1, k_max=2, k=self.k)
        self.blockrv = blockrv
        super(ParametricModelSpace, self).__init__(blockrv)

        # Positive constraint setup
        self.pos_constraint_ids = None
        self.pos_param_list = self.betaii_names + self.lambda_names

    def get_pos_constraint_ids(self):
        """
        Get parameter indices for positive constraints.

        Returns
        -------
        torch.Tensor
            Tensor containing indices of parameters that require positive constraints.
            These typically include diagonal factor loadings and noise variances.

        Notes
        -----
        Caches the result for efficiency on subsequent calls.
        """
        if self.pos_constraint_ids is not None:
            return self.pos_constraint_ids

        # Generate random variable dimension mapping
        rv_dim_dict = self.generateRVIndices()

        pos_param_id_list = []
        for param_name in self.pos_param_list:
            if param_name in rv_dim_dict:
                indices = rv_dim_dict[param_name]
                pos_param_id_list.append(torch.tensor(indices))

        # Concatenate all positive constraint indices
        if pos_param_id_list:
            self.pos_constraint_ids = torch.cat(pos_param_id_list)
        else:
            self.pos_constraint_ids = torch.tensor([], dtype=torch.long)
        return self.pos_constraint_ids

    def _to_tensor(self, data):
        """
        Convert input data to torch.Tensor with gradient support.

        Parameters
        ----------
        data : array-like
            Input data (torch.Tensor, numpy array, pandas DataFrame, or list)

        Returns
        -------
        torch.Tensor
            Converted tensor with float32 dtype and gradient preservation

        Notes
        -----
        Preserves gradient information for PyTorch automatic differentiation.
        Essential for variational inference and optimization algorithms.
        """
        if isinstance(data, torch.Tensor):
            return data
        elif hasattr(data, "values"):  # Handle pandas DataFrames/Series
            return torch.tensor(data.values, dtype=torch.float32)
        else:  # numpy array or list
            return torch.tensor(data, dtype=torch.float32)

    def log_prob(self, theta):
        """
        Compute total log probability (prior + likelihood) with gradient support.

        Parameters
        ----------
        theta : torch.Tensor, shape (n_samples, n_parameters)
            Parameter samples with requires_grad=True for gradient computation.

        Returns
        -------
        torch.Tensor, shape (n_samples,)
            Total log probability values with computational graph intact.

        Notes
        -----
        All operations preserve gradients for automatic differentiation.
        Essential for variational inference and gradient-based optimization.
        """
        return self.compute_prior(theta) + self.compute_llh(theta)

    def compute_prior(self, theta):
        """
        Compute log prior probability for all parameters.

        Parameters
        ----------
        theta : torch.Tensor, shape (n_samples, n_parameters)
            Parameter samples.

        Returns
        -------
        torch.Tensor, shape (n_samples,)
            Log prior probability values.

        Notes
        -----
        Prior specifications:
        - betaii (diagonal): HalfNormal(1) for positive constraint
        - betaij (off-diagonal): Normal(0, 1)
        - lambdaii (noise): InverseGamma(1.1, 0.05) for positive constraint

        All priors are chosen to be weakly informative and preserve gradient flow.
        """
        # Get parameter indices for all random variables
        cols = self.generateRVIndices()

        # Extract parameter indices for each group
        betaii_index = [item for name in self.betaii_names for item in cols[name]]
        betaij_index = [item for name in self.betaij_names for item in cols[name]]
        lambdaii_index = [item for name in self.lambda_names for item in cols[name]]

        # Extract parameter values (preserves gradients)
        betaii = theta[:, betaii_index]
        betaij = theta[:, betaij_index]
        lambdaii = theta[:, lambdaii_index]

        # Compute log prior probabilities
        betaii_prior_log_prob = HalfNormal(scale=torch.ones_like(betaii)).log_prob(betaii)
        betaij_prior_log_prob = Normal(loc=torch.zeros_like(betaij), scale=torch.ones_like(betaij)).log_prob(betaij)
        lambdaii_prior_log_prob = InverseGamma(concentration=1.1, rate=0.05).log_prob(lambdaii)

        # Sum over parameter dimensions
        betaii_prior_log_prob = torch.sum(betaii_prior_log_prob, dim=1)
        betaij_prior_log_prob = torch.sum(betaij_prior_log_prob, dim=1)
        lambdaii_prior_log_prob = torch.sum(lambdaii_prior_log_prob, dim=1)

        return betaii_prior_log_prob + betaij_prior_log_prob + lambdaii_prior_log_prob

    def compute_llh(self, theta):
        """
        Compute log-likelihood for factor analysis model.

        Parameters
        ----------
        theta : torch.Tensor, shape (n_samples, n_parameters)
            Parameter array.

        Returns
        -------
        llh : torch.Tensor, shape (n_samples,)
            Log-likelihood values.

        Notes
        -----
        The model assumes:
        - Lower triangular factor loading matrix W
        - Diagonal noise covariance matrix L
        - Covariance structure: Σ = WWᵀ + L
        - Multivariate normal likelihood for observed data
        - All computations preserve gradients for optimization

        Numerical stability is ensured by adding small diagonal regularization.
        """
        # Get parameter indices for all random variables
        cols = self.generateRVIndices()

        n = theta.shape[0]  # Number of parameter samples
        W = torch.zeros((n, self.obs_dim, self.k), device=theta.device)  # Factor loading matrix
        L = torch.zeros((n, self.obs_dim, self.obs_dim), device=theta.device)  # Noise covariance matrix

        # Construct factor loading matrix W (lower triangular)
        # Diagonal elements (positive constrained)
        for i, bn in enumerate(self.betaii_names):
            if len(cols[bn]) > 0:
                W[:, i, i] = theta[:, cols[bn]].flatten()

        # Off-diagonal elements (lower triangular part)
        j = 1
        i = 0
        for bn in self.betaij_names:
            if len(cols[bn]) > 0:
                W[:, j, i] = theta[:, cols[bn]].flatten()
            j += 1
            if j == self.obs_dim:
                i += 1
                j = i + 1

        # Construct noise covariance matrix L (diagonal, positive constrained)
        for i, ln in enumerate(self.lambda_names):
            if len(cols[ln]) > 0:
                L[:, i, i] = theta[:, cols[ln]].flatten()

        # Compute covariance matrix: Σ = WWᵀ + L
        cov = torch.einsum("...ij,...jk->...ik", W, W.transpose(-1, -2)) + L

        # Compute multivariate normal log-likelihood using torch.distributions
        log_likelihood = torch.zeros(n, device=theta.device)

        for i in range(cov.shape[0]):
            mvn = MultivariateNormal(loc=torch.zeros(self.obs_dim, device=theta.device), covariance_matrix=cov[i])
            log_likelihood[i] = mvn.log_prob(self.y_data).sum()

        return log_likelihood
