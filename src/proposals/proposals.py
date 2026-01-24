import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from .base import Proposal


class RWProposal(Proposal):
    """
    Random Walk proposal with optional Roberts-Rosenthal scaling.

    This proposal generates new samples by adding Gaussian noise to current
    parameters, with covariance estimated from the target distribution.
    Supports Roberts-Rosenthal optimal scaling for high-dimensional spaces.
    """

    def __init__(self, beta_names, rr_scale=False):
        """
        Initialize Random Walk proposal.

        Parameters
        ----------
        beta_names : list
            Names of parameters to be updated by this proposal
        rr_scale : bool, optional
            Whether to apply Roberts-Rosenthal scaling (default: False)
        """
        self.beta_names = beta_names
        self.rr = rr_scale
        super().__init__(beta_names)

    def calibratemmmpd(self, mmmpd, size, t):
        """
        Calibrate proposal using Multi-Model Mixture Posterior Density.

        Estimates covariance matrix from weighted particles and sets
        appropriate proposal scaling.

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
        theta, theta_w = mmmpd.getOriginalParticleDensityForTemperature(t, resample=False)

        model_key_indices, m_tuple_indices = self.getModel().enumerateModels(theta)

        if mk in model_key_indices.keys():
            mkidx = model_key_indices[mk]
            mk_theta = theta[mkidx]
            mk_theta_w = theta_w[mkidx]
            mk_theta_w = np.exp(mk_theta_w - logsumexp(mk_theta_w))
            X, concat_indices = self.concatParameters(mk_theta, mk, return_indices=True)

            self.d = X.shape[1]  # Parameter dimension
            self.cov = np.cov(X.T, aweights=mk_theta_w)  # Weighted covariance
        else:
            self.cov = np.eye(self.d)  # Default to identity matrix

        # Set proposal scale factor
        if self.rr:
            self.propscale = 2.38 / np.sqrt(self.d)  # Roberts-Rosenthal optimal scaling
        else:
            self.propscale = 1  # No additional scaling

    def draw(self, theta, size=1):
        """
        Generate proposals using random walk with calibrated covariance.

        Parameters
        ----------
        theta : ndarray
            Current parameter values
        size : int, optional
            Number of samples to draw (default: 1)

        Returns
        -------
        proptheta : ndarray
            Proposed parameter values
        zeros : ndarray
            Log proposal ratio (zero for symmetric proposals)
        ids : ndarray
            Proposal identifiers
        """
        mk = self.getModelIdentifier()
        N = theta.shape[0]
        X = self.concatParameters(theta, mk)
        d = X.shape[1]
        # Generate random walk proposals
        propX = X + multivariate_normal(np.zeros(d), self.cov * self.propscale).rvs(N)
        proptheta = self.deconcatParameters(propX, theta, mk)
        return proptheta, np.zeros(N), np.full(N, id(self))
