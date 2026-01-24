import numpy as np
from scipy.stats import halfnorm, invgamma, norm, poisson, rv_continuous


class Distribution:
    """
    Wrapper class for scipy statistical distributions.

    Provides a unified interface for various probability distributions
    with methods for sampling, density evaluation, and moment estimation.

    Notes
    -----
    - Wraps scipy.stats distributions with consistent API
    - Handles both continuous and discrete distributions
    - Supports parameter passing via kwargs
    """

    def __init__(self, distribution, **kwargs):
        """
        Initialize distribution wrapper.

        Parameters
        ----------
        distribution : scipy.stats distribution
            Scipy distribution object to wrap
        **kwargs : dict, optional
            Distribution parameters
        """
        self.dist = distribution
        self.kwargs = kwargs if kwargs is not None else {}
        self.dim = 1  # TODO: Support multi-dimensional distributions

    def draw(self, size=1):
        """
        Generate random samples from the distribution.

        Parameters
        ----------
        size : int, optional
            Number of samples to draw (default: 1)

        Returns
        -------
        ndarray
            Random samples from the distribution
        """
        return self.dist.rvs(**self.kwargs, size=size)

    def eval(self, theta):
        """
        Evaluate probability density/mass function.

        Parameters
        ----------
        theta : array_like
            Points at which to evaluate the distribution

        Returns
        -------
        ndarray
            Probability density/mass values
        """
        if isinstance(self.dist.dist, rv_continuous):
            return self.dist.pdf(theta, **self.kwargs)
        else:
            return self.dist.pmf(theta, **self.kwargs)

    def logeval(self, theta):
        """
        Evaluate log probability density/mass function.

        Parameters
        ----------
        theta : array_like
            Points at which to evaluate the distribution

        Returns
        -------
        ndarray
            Log probability density/mass values
        """
        if isinstance(self.dist.dist, rv_continuous):
            return self.dist.logpdf(theta, **self.kwargs)
        else:
            return self.dist.logpmf(theta, **self.kwargs)

    def _estimatemoments(self, theta, mk=None):
        """
        Estimate distribution moments from samples.

        Parameters
        ----------
        theta : ndarray
            Sample array for moment estimation
        mk : hashable, optional
            Model key (not used in base implementation)

        Returns
        -------
        mean : ndarray
            Estimated mean vector
        cov : ndarray
            Estimated covariance matrix
        """
        mean = np.mean(theta, axis=0)
        cov = np.cov(theta.T)
        return mean, cov


class InvGammaDistribution(Distribution):
    """
    Inverse Gamma distribution wrapper.

    The inverse gamma distribution is a two-parameter family of continuous
    probability distributions that is the reciprocal of the gamma distribution.
    """

    def __init__(self, a=1, b=1):
        """
        Initialize inverse gamma distribution.

        Parameters
        ----------
        a : float, optional
            Shape parameter (default: 1)
        b : float, optional
            Scale parameter (default: 1)
        """
        super().__init__(invgamma(a=a, scale=b))


class HalfNormalDistribution(Distribution):
    """
    Half-normal distribution wrapper.

    The half-normal distribution is a folded normal distribution restricted
    to non-negative values.
    """

    def __init__(self, sigma=1):
        """
        Initialize half-normal distribution.

        Parameters
        ----------
        sigma : float, optional
            Scale parameter (default: 1)
        """
        super().__init__(halfnorm(scale=sigma))


class NormalDistribution(Distribution):
    """
    Normal (Gaussian) distribution wrapper.

    The normal distribution is a continuous probability distribution
    characterized by its mean and standard deviation.
    """

    def __init__(self, mu=0, sigma=1):
        """
        Initialize normal distribution.

        Parameters
        ----------
        mu : float, optional
            Mean parameter (default: 0)
        sigma : float, optional
            Standard deviation (default: 1)
        """
        super().__init__(norm(mu, sigma))


class BoundedPoissonDistribution(Distribution):
    """
    Bounded Poisson distribution with rejection sampling.

    Poisson distribution restricted to a specified range [kmin, kmax]
    using rejection sampling.
    """

    def __init__(self, lam, kmin, kmax):
        """
        Initialize bounded Poisson distribution.

        Parameters
        ----------
        lam : float
            Rate parameter for Poisson distribution
        kmin : int
            Minimum allowed value (inclusive)
        kmax : int
            Maximum allowed value (inclusive)
        """
        self.kmin = kmin
        self.kmax = kmax
        super().__init__(poisson(lam))

    def draw(self, size=1):
        """
        Generate samples using rejection sampling.

        Parameters
        ----------
        size : int, optional
            Number of samples to draw (default: 1)

        Returns
        -------
        ndarray
            Samples from bounded Poisson distribution
        """
        # Rejection sampling implementation
        samples = np.zeros(size)
        draw_idx = np.full(size, True)
        while draw_idx.sum() > 0:
            samples[draw_idx] = self.dist.rvs(size=draw_idx.sum())
            draw_idx = np.logical_or(samples > self.kmax, samples < self.kmin)
        return samples


class ImproperDistribution(Distribution):
    """
    Improper (flat) distribution for non-informative priors.

    Represents an improper distribution with constant density.
    Sampling method is a placeholder and should be used with caution.
    """

    def __init__(self):
        """
        Initialize improper distribution.
        """
        self.dim = 1

    def draw(self, size=1):
        """
        Placeholder sampling method.

        Warning: This is a hack and does not represent proper sampling
        from an improper distribution.

        Parameters
        ----------
        size : int, optional
            Number of samples to draw (default: 1)

        Returns
        -------
        ndarray
            Standard normal samples (placeholder)
        """
        return norm(0, 1).rvs(size)

    def eval(self, theta):
        """
        Evaluate constant density function.

        Parameters
        ----------
        theta : array_like
            Points at which to evaluate (ignored)

        Returns
        -------
        ndarray
            Array of ones (constant density)
        """
        return np.ones(theta.shape)

    def logeval(self, theta):
        """
        Evaluate constant log density function.

        Parameters
        ----------
        theta : array_like
            Points at which to evaluate (ignored)

        Returns
        -------
        ndarray
            Array of zeros (constant log density)
        """
        return np.zeros(theta.shape)
