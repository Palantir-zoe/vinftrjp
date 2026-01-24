import sys

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from src.distributions import Distribution

from .base import Proposal


class IndependentProposal(Proposal):
    """
    Independent proposal that draws samples from a specified distribution.

    Generates proposals independently of current state, typically using
    prior distributions or fitted marginal distributions. Useful for
    independent Metropolis-Hastings or as components in mixture proposals.
    """

    def __init__(self, rv_name, proposal_distribution):
        """
        Initialize independent proposal.

        Parameters
        ----------
        rv_name : str
            Name of the random variable this proposal targets
        proposal_distribution : Distribution
            Distribution to draw independent samples from
        """
        assert isinstance(rv_name, str)
        super().__init__([rv_name])
        assert isinstance(proposal_distribution, Distribution)

        self.propdist = proposal_distribution

    def draw(self, theta, size=1):
        """
        Generate independent proposals from the specified distribution.

        Parameters
        ----------
        theta : ndarray
            Current parameter values (not used for independent proposals)
        size : int, optional
            Number of samples to draw (default: 1)

        Returns
        -------
        prop_theta : ndarray
            Independent proposals from proposal distribution
        prop_lpqratio : ndarray
            Log proposal ratio (currently zero, needs implementation)
        ids : ndarray
            Proposal identifiers
        """
        n = theta.shape[0]
        prop_lpqratio = np.zeros(n)
        return (
            self.propdist.draw(n),  # Draw independent samples
            prop_lpqratio,  # TODO: Compute proper proposal ratio
            np.full(n, id(self)),
        )


class MixtureProposal(Proposal):
    """
    Weighted mixture of multiple sub-proposals.

    Combines proposals from multiple sub-kernels using fixed weights.
    The mixture is computed by averaging the proposals and combining
    the proposal ratios appropriately.

    Notes
    -----
    - Weights must sum to 1.0
    - All sub-proposals operate on the entire parameter set
    - Final proposal is a weighted average of sub-proposal outputs
    """

    def __init__(self, subproposals, weights):
        """
        Initialize mixture proposal.

        Parameters
        ----------
        subproposals : list
            List of Proposal instances to mix
        weights : list
            List of mixture weights (must sum to 1.0)
        """
        assert isinstance(subproposals, list)
        assert isinstance(weights, list)
        assert len(weights) == len(subproposals)
        super().__init__(subproposals)

        # Validate and normalize weights
        wsum = 0.0
        for w in weights:
            assert isinstance(w, float)
            wsum += w
        assert wsum == 1  # Weights must sum to 1.0
        self.weights = np.array(weights)
        self.cumsumweights = np.cumsum(self.weights)

    def draw(self, theta, size=1):
        """
        Generate proposals by mixing outputs from all sub-proposals.

        Parameters
        ----------
        theta : ndarray
            Current parameter values
        size : int, optional
            Number of samples to draw (default: 1)

        Returns
        -------
        prop_theta : ndarray
            Weighted average of sub-proposal outputs
        prop_pqratio : ndarray
            Log of weighted average of proposal density ratios
        ids : ndarray
            Proposal identifiers
        """
        n = theta.shape[0]
        prop_theta = np.zeros_like(theta)
        prop_pqratio = np.zeros(n)

        # Combine proposals from all sub-kernels
        for i in range(len(self.ps)):
            pt, plq, ids = self.ps[i].draw(theta, n)
            prop_theta += self.weights[i] * pt  # Weighted average of proposals
            prop_pqratio += self.weights[i] * np.exp(plq)  # Weighted average in probability space

        return prop_theta, np.log(prop_pqratio), np.full(n, id(self))


class EigDecComponentwiseNormalProposal(Proposal):
    """
    Eigen-decomposed component-wise normal proposal with historical scaling.

    Uses eigenvalue decomposition and adjusts proposal scale based on
    historical acceptance rates rather than online optimization.

    Notes
    -----
    - No sub-proposals supported
    - Uses historical acceptance rates for scaling
    - More stable but less adaptive than trial version
    """

    def __init__(self, proposals=[]):
        """
        Initialize eigen-decomposed component-wise normal proposal.

        Parameters
        ----------
        proposals : list, optional
            List of sub-proposals (not used in this implementation)
        """
        self.propscale = 1.0
        super().__init__(proposals)
        self.dimension = 1
        self.propscale = 1

    def setAR(self, ar, t, n):
        """
        Update proposal scale based on historical acceptance rates.

        Parameters
        ----------
        ar : float
            Current acceptance rate
        t : int
            Timestamp/iteration
        n : int
            Number of proposals
        """
        super().setAR(ar, t, n)
        # Compute weighted average of recent acceptance rates
        if len(self.ar) < 10:
            avg_ar = 0.44  # Default acceptance rate
        else:
            alen = min(len(self.ar) - 1, 100)
            # Weighted average by proposal counts and timestamps
            avg_ar = np.sum(
                np.array(self.ar[-alen:]) * np.array(self.n_ar[-alen:]) * np.array(self.t_ar[-alen:])
            ) / np.sum(np.array(self.n_ar[-alen:]) * np.array(self.t_ar[-alen:]))

        # Update proposal scale using Robbins-Monro type adjustment
        self.propscale = np.exp(2.0 * self.dimension * (min(avg_ar, 0.7) - 0.44))
        print("eig[{}].propscale = {} for ar {}".format(self.dimension, self.propscale, avg_ar))

    def calibrate(self, theta, size, t):
        """
        Calibrate proposal using training samples.

        Estimates covariance matrix and performs eigenvalue decomposition.

        Parameters
        ----------
        theta : ndarray
            Training parameter samples
        size : int
            Sample size for calibration
        t : int
            Current iteration/timestamp
        """
        pmat = self.concatParameters(theta)
        m = pmat.shape[1]  # Parameter dimension
        self.dimension = m
        n = theta.shape[0]
        cov = np.cov(pmat.T)  # Estimate covariance matrix

        # Handle scalar covariance case
        if len(cov.shape) == 0:
            self.valid_indices = np.arange(1)
            self.num_valid_indices = 1
            if not np.isfinite(cov):
                cov = 0.5
            self.eigvals = cov

        # Multi-dimensional case
        else:
            m = cov.shape[0]
            # Ensure covariance matrix is finite
            if not np.isfinite(cov).all():
                cov = np.eye(m) * 0.5

            # Regularize covariance matrix if near-singular
            while np.linalg.det(cov) < 0.001:
                cov += np.eye(m) * 0.001 * (max(1 - np.linalg.det(cov), 0))

            try:
                # Perform eigenvalue decomposition
                self.eigvals, self.eigvecs = np.linalg.eigh(cov)
            except np.linalg.LinAlgError:
                print("Linalg error in eigenvalue decomposition")
                print(cov)
                sys.exit(0)

            self.valid_indices = np.arange(self.eigvals.shape[0])
            self.num_valid_indices = self.valid_indices.shape[0]

    def draw(self, theta, size=1):
        """
        Generate proposals using eigen-decomposed component-wise updates.

        Parameters
        ----------
        theta : ndarray
            Current parameter values
        size : int, optional
            Number of samples to draw (default: 1)

        Returns
        -------
        prop_theta : ndarray
            Proposed parameter values
        zeros : ndarray
            Log proposal ratio (zero for symmetric proposals)
        ids : ndarray
            Proposal identifiers
        """
        pmat = self.concatParameters(theta)
        n = pmat.shape[0]
        m = pmat.shape[1]

        # Single parameter case
        if m == 1:
            prop_pmat = pmat + norm(0, np.sqrt(self.eigvals) * self.propscale).rvs(n)
        # Multi-parameter case
        else:
            # Randomly select components to update
            ii = np.random.randint(self.num_valid_indices, size=n)
            i = self.valid_indices[ii]
            # Transform to eigenbasis
            pmat_r = np.einsum("ij,kj->ki", self.eigvecs.T, pmat)
            # Add noise along selected eigenvectors
            pmat_r[(np.arange(n), i)] += norm(0, np.sqrt(np.abs(self.eigvals[i])) * self.propscale).rvs(n)
            # Transform back to original basis
            prop_pmat = np.einsum("ij,kj->ki", self.eigvecs, pmat_r)

        prop_theta = self.deconcatParameters(prop_pmat, theta)
        return prop_theta, np.zeros(n), np.full(n, id(self))


class EigDecComponentwiseNormalProposalTrial(Proposal):
    """
    Eigen-decomposed component-wise normal proposal with adaptive scaling.

    Uses eigenvalue decomposition of the covariance matrix to propose updates
    along principal components. Includes automatic proposal scaling to achieve
    target acceptance rate.

    Notes
    -----
    - No sub-proposals supported
    - Uses eigenvalue decomposition for efficient sampling
    - Automatically adjusts proposal scale to target acceptance rate
    """

    def __init__(self, proposals=[]):
        """
        Initialize eigen-decomposed component-wise normal proposal.

        Parameters
        ----------
        proposals : list, optional
            List of sub-proposals (not used in this implementation)
        """
        self.propscale = 1.0
        super().__init__(proposals)
        self.dimension = 1
        self.propscale = 1

    def calibratemmmpd(self, mmmpd, size, t):
        """
        Calibrate proposal using Multi-Model Mixture Posterior Density.

        Estimates covariance matrix, performs eigenvalue decomposition,
        and optimizes proposal scale to target acceptance rate.

        Parameters
        ----------
        mmmpd : object
            MultiModelMPD object containing mixture posterior densities
        size : int
            Sample size for calibration
        t : int
            Current iteration/timestamp
        """
        mk = self.getModelIdentifier()
        theta, theta_w = mmmpd.getParticleDensityForModelAndTemperature(mk, t, resample=True, resample_max_size=500)

        pmat = self.concatParameters(theta, mk)

        m = pmat.shape[1]  # Parameter dimension
        self.dimension = m

        n = pmat.shape[0]
        cov = np.cov(pmat.T)  # Estimate covariance matrix

        # Handle scalar covariance case
        if len(cov.shape) == 0:
            self.valid_indices = np.arange(1)  # Single parameter
            self.num_valid_indices = 1
            if not np.isfinite(cov):
                cov = 0.5  # Default variance for numerical stability
            self.eigvals = cov  # Use variance directly for scaling

        # Handle zero-dimensional case
        elif cov.shape[0] == 0:
            self.valid_indices = np.arange(0)  # No parameters
            self.num_valid_indices = 0
            self.eigvals = 0
            return

        # Multi-dimensional case
        else:
            m = cov.shape[0]
            # Ensure covariance matrix is finite and well-conditioned
            if not np.isfinite(cov).all():
                cov = np.eye(m) * 0.5

            # Regularize covariance matrix if near-singular
            while np.linalg.det(cov) < 0.001:
                cov += np.eye(m) * 0.001 * (max(1 - np.linalg.det(cov), 0))

            try:
                # Perform eigenvalue decomposition
                self.eigvals, self.eigvecs = np.linalg.eigh(cov)
            except np.linalg.LinAlgError:
                print("Linalg error in eigenvalue decomposition")
                print(cov)
                sys.exit(0)

            self.valid_indices = np.arange(self.eigvals.shape[0])
            self.num_valid_indices = self.valid_indices.shape[0]

            # Optimize proposal scale to target acceptance rate
            target_ar = 0.234  # Optimal acceptance rate for random walk
            toler = 0.0001  # Tolerance for convergence
            maxiter = 20  # Maximum iterations
            avg_ar = target_ar
            self.propscale = 0.1  # Initial proposal scale

            # Sample subset for acceptance rate estimation
            ssize = 64
            ids = np.random.choice(pmat.shape[0], ssize)
            llh = self.pmodel.compute_llh(theta[ids])
            cur_prior = self.pmodel.compute_prior(theta[ids])

            def get_ar():
                """Estimate current acceptance rate."""
                prop_theta, prop_lpqratio, prop_id = self.draw(theta[ids])
                prop_prior = self.pmodel.compute_prior(prop_theta)
                prop_llh = self.pmodel.compute_llh(prop_theta)
                log_ar = self.pmodel.compute_lar(
                    theta[ids],
                    prop_theta,
                    np.zeros(ssize),
                    prop_llh,
                    llh,
                    cur_prior,
                    prop_prior,
                    t,
                )
                ar = np.exp(logsumexp(log_ar) - np.log(log_ar.shape[0]))
                return ar

            def get_propscale(ar):
                """Update proposal scale based on acceptance rate."""
                return np.exp(2.0 * self.dimension * (ar - target_ar))

            # Binary search for optimal proposal scale
            lb = 0.0001  # Lower bound
            ub = 2.0  # Upper bound
            next_ps = 0.5 * (lb + ub)
            ps = ub

            while maxiter > 0:
                if abs(ps - next_ps) < toler:
                    break
                ps = next_ps
                self.propscale = ps
                ar = get_ar()
                # Adjust bounds based on acceptance rate
                if ar < target_ar:
                    ub = ps  # Reduce scale if acceptance too low
                else:
                    lb = ps  # Increase scale if acceptance too high
                next_ps = 0.5 * (lb + ub)
                maxiter -= 1

    def draw(self, theta, size=1):
        """
        Generate proposals using eigen-decomposed component-wise updates.

        Parameters
        ----------
        theta : ndarray
            Current parameter values
        size : int, optional
            Number of samples to draw (default: 1)

        Returns
        -------
        prop_theta : ndarray
            Proposed parameter values
        zeros : ndarray
            Log proposal ratio (zero for symmetric proposals)
        ids : ndarray
            Proposal identifiers
        """
        pmat = self.concatParameters(theta, self.getModelIdentifier())
        n = pmat.shape[0]
        m = pmat.shape[1]

        # Handle zero-dimensional case
        if self.num_valid_indices == 0:
            return theta, np.zeros(n), np.full(n, id(self))

        # Single parameter case
        if m == 1:
            noise = norm(0, np.sqrt(self.eigvals) * self.propscale).rvs(n)
            prop_pmat = pmat + noise.reshape(pmat.shape)
        # Multi-parameter case
        else:
            # Randomly select components to update
            ii = np.random.randint(self.num_valid_indices, size=n)
            i = self.valid_indices[ii]
            # Transform to eigenbasis
            pmat_r = np.einsum("ij,kj->ki", self.eigvecs.T, pmat)
            # Add noise along selected eigenvectors
            pmat_r[(np.arange(n), i)] += norm(0, np.sqrt(np.abs(self.eigvals[i])) * self.propscale).rvs(n)
            # Transform back to original basis
            prop_pmat = np.einsum("ij,kj->ki", self.eigvecs, pmat_r)

        prop_theta = self.deconcatParameters(prop_pmat, theta, self.getModelIdentifier())
        return prop_theta, np.zeros(n), np.full(n, id(self))
