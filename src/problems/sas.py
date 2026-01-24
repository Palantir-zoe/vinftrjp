import numpy as np
import torch
from nflows.distributions.normal import StandardNormal
from torch import nn

from src.flows import Flow
from src.transforms import (
    CompositeTransform,
    InverseTransform,
    LTransform,
    SAS2DTransform,
    SinArcSinhTransform,
)

from .problem import Problem


class SAS(Problem):
    """
    Unified interface for SinArcSinh transformed distributions.

    This class provides a unified interface to access both 1D and 2D
    SinArcSinh transformed distributions, combining them into a single
    module for convenient usage. The SinArcSinh transformation allows
    for flexible modeling of skewed and heavy-tailed distributions.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments for configuration:
        - ep: epsilon parameters for transformation [ε1, ε2, ε3]
        - dp: delta parameters for transformation [δ1, δ2, δ3]
        - m1prob: mixture probability for first component

    Attributes
    ----------
    ep : numpy.ndarray
        Epsilon parameters controlling skewness of the transformation
    dp : numpy.ndarray
        Delta parameters controlling tail behavior of the transformation
    m1prob : float
        Mixture probability for the first component in mixture distributions
    ndim : int or None
        Dimensionality of the target distribution (set when target is created)

    Notes
    -----
    The SinArcSinh transformation is defined as:
        T(x) = sinh(δ * arcsinh(x) - ε)
    where:
        - ε (epsilon) controls skewness
        - δ (delta) controls tail behavior
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the SinArcSinh distribution interface.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for configuration:
            - ep: epsilon parameters for transformation [ε1, ε2, ε3]
            - dp: delta parameters for transformation [δ1, δ2, δ3]
            - m1prob: mixture probability for first component
        """
        super().__init__()

        # Set epsilon parameters (skewness parameters)
        # Default values provide a range of skewness patterns
        ep = kwargs.get("ep")
        if ep is None:
            ep = np.array([-2, 1.5, -2], dtype=np.float64)
        self.ep = ep

        # Set delta parameters (tail parameters)
        # Default values provide varying tail thickness
        dp = kwargs.get("dp")
        if dp is None:
            dp = np.array([1.0, 1.0, 1.5], dtype=np.float64)
        self.dp = dp

        # Set mixture probability for multi-component distributions
        m1prob = kwargs.get("m1prob")
        if m1prob is None:
            m1prob = 0.25
        self.m1prob = m1prob

        # Dimensionality will be set when target distribution is created
        self.ndim = None

    def target(self, k, **kwargs):
        """
        Create target distribution based on dimensionality.

        Parameters
        ----------
        k : int
            Dimensionality selector:
            - 0: 1D SinArcSinh transformed normal distribution
            - 1: 2D correlated SinArcSinh transformed bivariate normal distribution
        **kwargs : dict
            Additional keyword arguments passed to the target distribution

        Returns
        -------
        SASk0D1 or SASk1D2
            Target distribution instance with specified dimensionality

        Raises
        ------
        ValueError
            If k is not 0 or 1

        Examples
        --------
        >>> sas = SAS()
        >>> target_1d = sas.target(0)  # 1D distribution
        >>> target_2d = sas.target(1)  # 2D distribution

        Notes
        -----
        The dimensionality parameter k allows switching between different
        distribution types without changing the interface, making it
        convenient for model comparison and testing.
        """
        if k == 0:
            # Create 1D SinArcSinh transformed normal distribution
            target_ = SASk0D1(ep=self.ep, dp=self.dp, m1prob=self.m1prob, **kwargs)
        elif k == 1:
            # Create 2D correlated SinArcSinh transformed bivariate normal distribution
            target_ = SASk1D2(ep=self.ep, dp=self.dp, m1prob=self.m1prob, **kwargs)
        else:
            raise ValueError(f"k must be 0 or 1 not {k}.")

        # Store the dimensionality of the created target distribution
        self.ndim = target_.ndim

        return target_


class SASbase(nn.Module):
    """
    Base class for SinArcSinh transformed distributions.

    Parameters
    ----------
    ndim : int
        Number of dimensions of the distribution
    ep : np.ndarray, optional
        Epsilon parameters for transformations, by default [-2, 1.5, -2]
    dp : np.ndarray, optional
        Delta parameters for transformations, by default [1.0, 1.0, 1.5]
    m1prob : float, optional
        Mixture probability for first component, by default 0.25
    """

    def __init__(self, ndim, ep=None, dp=None, m1prob=None) -> None:
        super().__init__()
        self.ndim = ndim

        if ep is None:
            ep = np.array([-2, 1.5, -2], dtype=np.float64)
        self.ep = ep

        if dp is None:
            dp = np.array([1.0, 1.0, 1.5], dtype=np.float64)
        self.dp = dp

        if m1prob is None:
            m1prob = 0.25
        self.m1prob = m1prob

        self.flow: Flow = self._inlitialize_flow()

    def _inlitialize_flow(self) -> Flow:
        raise NotImplementedError

    def log_prob(self, z):
        """
        Compute log probability of the distribution.

        Parameters
        ----------
        z : torch.Tensor
            Input tensor of shape (N, self.ndim)

        Returns
        -------
        torch.Tensor
            Log probabilities of shape (N,)
        """
        # z: (N, 1)
        if z.ndim == 1:
            z = z.unsqueeze(-1)
        # z: (N, 2)
        return self.flow.log_prob(z).squeeze(-1) + np.log(1 - self.m1prob)

    def sample(self, num_samples=1):
        """
        Generate samples from the distribution.

        Parameters
        ----------
        num_samples : int, optional
            Number of samples to generate, by default 1

        Returns
        -------
        torch.Tensor
            Generated samples from the distribution
        """
        return self.flow.sample(num_samples)


class SASk0D1(SASbase):
    """
    Target for k=0: 1D SinArcSinh-transformed N(0,1).

    Notes
    -----
    A one-dimensional normal distribution transformed by SinArcSinh transformation.
    This represents the first component in a mixture distribution.
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs
            Additional arguments passed to SASbase
        """
        super().__init__(ndim=1, **kwargs)

    def _inlitialize_flow(self) -> Flow:
        self.tf = InverseTransform(SinArcSinhTransform(self.ep[0], self.dp[0]))
        self.base = StandardNormal((1,))

        return Flow(self.tf, self.base)


class SASk1D2(SASbase):
    """
    Target for k=1: 2D correlated SinArcSinh-transformed bivariate normal.

    Notes
    -----
    A two-dimensional correlated normal distribution transformed by SinArcSinh transformation.
    This represents the second component in a mixture distribution.
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs
            Additional arguments passed to SASbase
        """
        super().__init__(ndim=2, **kwargs)

    def _inlitialize_flow(self) -> Flow:
        L = torch.linalg.cholesky(torch.tensor([[1.0, 0.99], [0.99, 1.0]]))
        self.tf = InverseTransform(
            CompositeTransform([LTransform(L), SAS2DTransform([self.ep[1], self.ep[2]], [self.dp[1], self.dp[2]])])
        )
        self.base = StandardNormal((2,))

        return Flow(self.tf, self.base)
