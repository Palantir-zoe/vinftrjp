import numpy as np
from scipy.stats import norm, randint

from src.distributions import (
    BoundedPoissonDistribution,
    Distribution,
    HalfNormalDistribution,
    ImproperDistribution,
    InvGammaDistribution,
    NormalDistribution,
)

# from src.utils.linalgtools import *

np.set_printoptions(linewidth=200)


class RandomVariable:
    """
    Base class for random variables in probabilistic models.

    Encapsulates a random variable with its prior distribution and provides
    methods for sampling and density evaluation.

    Notes
    -----
    - Serves as base class for all random variable types
    - Prior distribution completely defines the random variable
    - Supports both proper and improper priors
    """

    def __init__(self, prior_distribution=Distribution(norm(0, 1))):
        """
        Initialize random variable with prior distribution.

        Parameters
        ----------
        prior_distribution : Distribution, optional
            Prior distribution for the random variable (default: standard normal)
        """
        self.priordist = prior_distribution
        # I don't think there is anything else... the prior completely defines the RV.

    def dim(self, model_key=None):
        """
        Get the dimension of the random variable.

        Parameters
        ----------
        model_key : hashable, optional
            Model identifier for trans-dimensional cases

        Returns
        -------
        int
            Dimension of the random variable
        """
        return self.priordist.dim

    def setModel(self, m):
        """
        Set the probabilistic model for this random variable.

        Parameters
        ----------
        m : object
            Probabilistic model instance
        """
        self.pmodel = m

    def getModel(self):
        """
        Get the associated probabilistic model.

        Returns
        -------
        object
            Probabilistic model instance
        """
        return self.pmodel

    def getModelIdentifier(self):
        """
        Get model identifier columns.

        Returns
        -------
        None
            Base implementation returns None. Override for trans-dimensional cases.
        """
        return None

    def getRange(self):
        """
        Get the range of possible values.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses
        """
        raise NotImplementedError

    def draw(self, size=1):
        """
        Draw samples from the prior distribution.

        Parameters
        ----------
        size : int, optional
            Number of samples to draw (default: 1)

        Returns
        -------
        ndarray
            Samples from the prior distribution
        """
        return self.priordist.draw(size)

    def eval_log_prior(self, theta):
        """
        Evaluate log prior density at given parameter values.

        Parameters
        ----------
        theta : ndarray
            Parameter values at which to evaluate log prior

        Returns
        -------
        ndarray
            Log prior density values
        """
        # return np.log(self.priordist.eval(theta))
        return self.priordist.logeval(theta)

    def _estimatemoments(self, theta, mk=None):
        """
        Estimate moments from samples.

        Parameters
        ----------
        theta : ndarray
            Sample array for moment estimation
        mk : hashable, optional
            Model key identifier

        Returns
        -------
        tuple
            Estimated mean and covariance
        """
        # raise "Not implemented"
        return self.priordist._estimatemoments(theta, mk)


class ImproperRV(RandomVariable):
    """
    Random variable with improper (flat) prior distribution.

    Useful for non-informative priors in Bayesian analysis.
    """

    def __init__(self):
        """
        Initialize improper random variable.
        """
        super().__init__(prior_distribution=ImproperDistribution())


class NormalRV(RandomVariable):
    """
    Normally distributed random variable.

    Gaussian random variable with specified mean and standard deviation.
    """

    def __init__(self, mu=0, sigma=1):
        """
        Initialize normal random variable.

        Parameters
        ----------
        mu : float, optional
            Mean parameter (default: 0)
        sigma : float, optional
            Standard deviation (default: 1)
        """
        super().__init__(prior_distribution=NormalDistribution(mu, sigma))


class HalfNormalRV(RandomVariable):
    """
    Half-normally distributed random variable.

    Folded normal distribution restricted to non-negative values.
    """

    def __init__(self, sigma=1):
        """
        Initialize half-normal random variable.

        Parameters
        ----------
        sigma : float, optional
            Scale parameter (default: 1)
        """
        super().__init__(prior_distribution=HalfNormalDistribution(sigma))


class InvGammaRV(RandomVariable):
    """
    Inverse Gamma distributed random variable.

    Useful as conjugate prior for variance parameters in normal models.
    """

    def __init__(self, a=1, b=1):
        """
        Initialize inverse gamma random variable.

        Parameters
        ----------
        a : float, optional
            Shape parameter (default: 1)
        b : float, optional
            Scale parameter (default: 1)
        """
        super().__init__(prior_distribution=InvGammaDistribution(a, b))


class UniformIntegerRV(RandomVariable):
    """
    Uniformly distributed integer random variable.

    Discrete uniform distribution over specified integer range.
    """

    def __init__(self, imin, imax):
        """
        Initialize uniform integer random variable.

        Parameters
        ----------
        imin : int
            Minimum value (inclusive)
        imax : int
            Maximum value (inclusive)
        """
        self.imin = imin
        self.imax = imax  # inclusive
        super().__init__(prior_distribution=Distribution(randint(low=imin, high=imax + 1)))

    def getRange(self):
        """
        Get the range of possible integer values.

        Returns
        -------
        list
            List of integers from imin to imax (inclusive)
        """
        return list(range(self.imin, self.imax + 1))


class BoundedPoissonRV(RandomVariable):
    """
    Bounded Poisson distributed random variable.

    Poisson distribution restricted to a specified integer range [imin, imax].
    Uses rejection sampling to generate samples within bounds.

    Notes
    -----
    - Extends Poisson distribution with bounds constraints
    - Useful for count data with known minimum and maximum values
    - Implements rejection sampling for bounded sampling
    """

    def __init__(self, lam, imin, imax):
        """
        Initialize bounded Poisson random variable.

        Parameters
        ----------
        lam : float
            Rate parameter for Poisson distribution
        imin : int
            Minimum allowed value (inclusive)
        imax : int
            Maximum allowed value (inclusive)
        """
        self.lam = lam
        self.imin = imin
        self.imax = imax
        super().__init__(prior_distribution=BoundedPoissonDistribution(lam, imin, imax))

    def getRange(self):
        """
        Get the range of possible integer values.

        Returns
        -------
        list
            List of integers from imin to imax (inclusive)
        """
        return list(range(self.imin, self.imax + 1))
