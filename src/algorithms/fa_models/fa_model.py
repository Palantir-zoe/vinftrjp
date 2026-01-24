import numpy as np

from src.variables import (
    ConditionalVariableBlock,
    HalfNormalRV,
    InvGammaRV,
    NormalRV,
    ParametricModelSpace,
    UniformIntegerRV,
)


class FactorAnalysisModel(ParametricModelSpace):
    """
    Factor Analysis model with conditional variable structure.

    Implements a Bayesian factor analysis model with lower-triangular factor
    loading matrix and diagonal noise covariance. Supports variable number
    of factors through conditional variable blocks.

    Notes
    -----
    - Uses lower-triangular constraint for identifiability
    - Diagonal elements (beta_ii) follow half-normal distribution
    - Off-diagonal elements (beta_ij) follow normal distribution
    - Noise variances (lambda_i) follow inverse gamma distribution
    - Supports reversible jump between different numbers of factors
    """

    def __init__(self, problem, y_data, *, k_min=1, k_max=2, k=3, **kwargs):
        """
        Initialize factor analysis model.

        Parameters
        ----------
        y_data : ndarray
            Observed data matrix of shape (n_observations, n_features)
        k_min : int, optional
            Minimum number of factors (default: 1)
        k_max : int, optional
            Maximum number of factors (default: 2)
        k : int, optional
            The number of factors (default: 3)
        obs_dim : int, optional
            Observation dimension (default: inferred from y_data)
        **kwargs : dict
            Additional keyword arguments for model configuration
        """
        self.problem = problem
        self.y_data = y_data

        self.k = k
        self.obs_dim = y_data.shape[1]

        self.betaii_names = []  # Diagonal element names
        self.betaij_names = []  # Off-diagonal element names
        self.lambda_names = []  # Noise variance names

        self.random_variables, _ = self._setup_random_variables(k_min, k_max, **kwargs)
        self.proposal = self._setup_proposal(k_min, k_max, **kwargs)

        super().__init__(self.random_variables, self.proposal)

    def _setup_random_variables(self, k_min, k_max, **kwargs):
        """
        Setup random variables for the factor analysis model.

        Parameters
        ----------
        k_min : int
            Minimum number of factors
        k_max : int
            Maximum number of factors
        **kwargs : dict
            Additional configuration parameters

        Returns
        -------
        random_variables : dict
            Dictionary of random variables for the model space
        """
        blockrv = {}
        self.blockrvcond = {}

        # Setup diagonal elements (beta_ii) with half-normal prior
        for col in range(0, self.k):
            name = f"beta{col}{col}"
            self.betaii_names.append(name)
            blockrv[name] = HalfNormalRV(1)  # Half-normal with scale=1
            self.blockrvcond[name] = col  # Activation condition

            # Setup off-diagonal elements (beta_ij) with normal prior
            for row in range(col + 1, self.obs_dim):
                name = f"beta{row}{col}"
                self.betaij_names.append(name)
                blockrv[name] = NormalRV(0, 1)  # Normal with mean=0, std=1
                self.blockrvcond[name] = col  # Activation condition

        random_variables = {}

        # Create conditional variable block for factor loadings
        name = "allbeta"
        random_variables[name] = ConditionalVariableBlock(
            blockrv, self.blockrvcond, UniformIntegerRV(k_min, k_max), "k"
        )

        # Setup noise variances with inverse gamma prior
        for i in range(self.obs_dim):
            name = f"lambda{i}"
            self.lambda_names.append(name)
            rv = InvGammaRV(1.1, 0.05)  # Inverse gamma with a=1.1, b=0.05
            random_variables[name] = rv
            blockrv[name] = rv

        return random_variables, blockrv

    def _setup_proposal(self, k_min, k_max, **kwargs):
        """
        Setup proposal distributions for MCMC sampling.

        Parameters
        ----------
        k_min : int
            Minimum number of factors
        k_max : int
            Maximum number of factors
        **kwargs : dict
            Additional configuration parameters

        Returns
        -------
        Proposal
            Proposal object for MCMC sampling

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses
        """
        raise NotImplementedError

    def sanitise(self, inputs):
        """
        Sanitize parameter values by enforcing conditional constraints.

        Sets beta parameters to zero when their activation conditions
        are not met based on the current number of factors.

        Parameters
        ----------
        inputs : ndarray
            Parameter array to sanitize

        Returns
        -------
        outputs : ndarray
            Sanitized parameter array with proper zero constraints
        """
        # need to set betas to zero when not in block
        outputs = inputs.copy()
        mkdict, _ = self.enumerateModels(inputs)

        # Apply conditional constraints for each model type
        for mk, idx in mkdict.items():
            tn = idx.shape[0]  # Number of samples for this model

            # Zero out beta parameters that should not be active
            for nm in self.betaii_names + self.betaij_names:
                if mk[0] < self.blockrvcond[nm]:
                    outputs[idx] = self.proposal.setVariable(outputs[idx], nm, np.zeros(tn))
        return outputs

    def compute_llh(self, theta):
        """
        Compute log-likelihood for factor analysis model.

        Parameters
        ----------
        theta : ndarray
            Parameter array of shape (n_samples, n_parameters)

        Returns
        -------
        llh : ndarray
            Log-likelihood values of shape (n_samples,)

        Notes
        -----
        The model assumes:
        - Lower triangular factor loading matrix W
        - Diagonal noise covariance matrix L
        - Covariance structure: Σ = WWᵀ + L
        - Multivariate normal likelihood for observed data
        """
        y_data = self.y_data

        # Get parameter indices for all random variables
        cols = self.generateRVIndices()

        n = theta.shape[0]  # Number of parameter samples
        W = np.zeros((n, self.obs_dim, self.k))  # Factor loading matrix
        L = np.zeros((n, self.obs_dim, self.obs_dim))  # Noise covariance matrix

        # Construct factor loading matrix W (lower triangular)
        # Diagonal elements
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

        # Construct noise covariance matrix L (diagonal)
        for i, ln in enumerate(self.lambda_names):
            if len(cols[ln]) > 0:
                L[:, i, i] = theta[:, cols[ln]].flatten()

        # Compute covariance matrix: Σ = WWᵀ + L
        cov = np.einsum("...ij,...jk", W, np.einsum("...ji", W)) + L

        # Compute log-determinant of covariance matrix
        _, logdets = np.linalg.slogdet(cov)

        # Compute inverse of covariance matrix
        invs = np.linalg.inv(cov)

        # Compute multivariate normal log-likelihood
        return -0.5 * (
            y_data.shape[0] * (logdets + 6 * np.log(2 * np.pi))
            + np.einsum("i...j,i...j", np.einsum("...k,ijk", y_data, invs), y_data)
        )
