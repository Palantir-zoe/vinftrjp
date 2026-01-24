import numpy as np
import torch
from nflows.distributions.normal import StandardNormal

from src.flows import Flow
from src.transforms import (
    CompositeTransform,
    InverseTransform,
    LTransform,
    SAS2DTransform,
    SinArcSinhTransform,
)
from src.variables import (
    ImproperRV,
    ParametricModelSpace,
    TransDimensionalBlock,
)


class ToyModel(ParametricModelSpace):
    """
    Toy model for testing reversible jump MCMC with SinArcSinh transformed distributions.

    This class implements a trans-dimensional model space with two competing models:
    - Model 0: 1D SinArcSinh transformed normal distribution
    - Model 1: 2D correlated SinArcSinh transformed bivariate normal distribution

    Parameters
    ----------
    proposal_class : class
        Class for creating reversible jump proposals
    problem :
        instance of class Problem
    """

    def __init__(self, proposal_class, problem, **kwargs):
        self.proposal_class = proposal_class
        self.problem = problem

        self.verbose = kwargs.get("verbose", False)
        self.run_index = kwargs.get("run_index", None)

        self.random_variables = self._setup_random_variables()
        self.proposal = self._setup_proposal(**kwargs)

        super().__init__(self.random_variables, self.proposal)

    def _setup_random_variables(self, **kwargs):
        """
        Setup random variables for the model space.

        Returns
        -------
            Random variables dictionary
        """
        random_variables = {
            "t1": ImproperRV(),
            "block2": TransDimensionalBlock({"t2": ImproperRV()}, nblocks_name="k", minimum_blocks=0, maximum_blocks=1),
        }

        return random_variables

    def _setup_proposal(self, **kwargs):
        """
        Setup proposal distributions for the model space.

        Returns
        -------
            proposal object
        """
        raise NotImplementedError

    def sas(self, x, i):
        """
        Apply SinArcSinh transformation to input values.

        Parameters
        ----------
        x : array_like
            Input values to transform
        i : int
            Index specifying which transformation parameters to use

        Returns
        -------
        array_like
            Transformed values

        Notes
        -----
        TODO: Compute Jacobian for each transformation. Parameters are not dependent.
        """

        def _sas(x, epsilon, delta):
            return np.sinh((np.arcsinh(x) + epsilon) / delta)

        epsilon = np.array([-2, 2, -2, 2, -2])
        delta = np.array([1, 1, 1, 1, 1])
        return _sas(x, epsilon[i], delta[i])

    def draw_perfect(self, M):
        """
        Generate perfect samples from the target distribution.

        Parameters
        ----------
        M : int
            Number of samples to generate

        Returns
        -------
        numpy.ndarray
            Array of samples with shape (M, self.dim())

        Notes
        -----
        This method generates samples by transforming samples from the base
        normal distribution using the inverse flow transformations.
        Model indicators are drawn from the prior distribution.
        """
        cols = self.generateRVIndices()

        # draw it from prior first
        theta = np.zeros((M, self.dim()))
        # hack it
        theta[:] = self.sampleFromPrior(M)
        # print(theta)
        # That sorts out model indicators for now.
        # FIXME for now we leave model indicators as drawn from prior
        # and just draw perfect draws for these models again.
        # This doesn't give us a complete joint posterior
        # because the model probability marginals are wrong, they're the prior.
        # For calibration, we don't care.

        k = theta[:, cols["k"]].flatten()
        t1 = theta[:, cols["t1"]].flatten()
        t2 = theta[:, cols["t2"]].flatten()
        m1 = k == 0
        m2 = k == 1

        tf1 = InverseTransform(SinArcSinhTransform(self.problem.ep[0], self.problem.dp[0]))
        tf2 = InverseTransform(
            CompositeTransform(
                [
                    LTransform(torch.linalg.cholesky(torch.tensor([[1.0, 0.99], [0.99, 1.0]]))),
                    SAS2DTransform(self.problem.ep[1:3], self.problem.dp[1:3]),
                ]
            )
        )

        f1 = Flow(transform=tf1, distribution=StandardNormal((1,)))
        f2 = Flow(transform=tf2, distribution=StandardNormal((2,)))
        m1_v = f1.sample(t1[m1].shape[0]).detach().numpy()
        t1[m1] = m1_v[:, 0]
        m2_v = f2.sample(t1[m2].shape[0]).detach().numpy()
        t1[m2] = m2_v[:, 0]
        t2[m2] = m2_v[:, 1]  # TODO, Why do like this?

        theta[:, cols["t1"]] = t1.reshape(theta[:, cols["t1"]].shape)
        theta[:, cols["t2"]] = t2.reshape(theta[:, cols["t2"]].shape)
        return theta

    def compute_llh(self, theta):
        r"""
        Compute log-likelihood for given parameters.

        Parameters
        ----------
        theta : numpy.ndarray
            Parameter array with shape (n_samples, n_parameters)

        Returns
        -------
        numpy.ndarray
            Log-likelihood values for each sample

        Notes
        -----
        Uses the RJ target from Andrieu et al 2009:
        $\pi(\theta,k) = 0.25 * N(\theta;0,1) * I(k==1) + 0.75 * N(\theta;[0,0],[[1,-0.9],[-0.9,1]]) * I(k==2)$
        where $k \in {1,2}$
        """
        cols = self.generateRVIndices()
        k = theta[:, cols["k"]].flatten()
        t1 = theta[:, cols["t1"]].flatten()
        t2 = theta[:, cols["t2"]].flatten()
        m1 = k == 0
        m2 = k == 1
        llh = np.zeros(k.shape[0])

        tf1 = InverseTransform(SinArcSinhTransform(self.problem.ep[0], self.problem.dp[0]))
        tf2 = InverseTransform(
            CompositeTransform(
                [
                    LTransform(torch.linalg.cholesky(torch.tensor([[1.0, 0.99], [0.99, 1.0]]))),
                    SAS2DTransform(self.problem.ep[1:3], self.problem.dp[1:3]),
                ]
            )
        )
        f1 = Flow(transform=tf1, distribution=StandardNormal((1,)))
        f2 = Flow(transform=tf2, distribution=StandardNormal((2,)))
        if m1.sum() > 0:
            llh[m1] = (
                np.log(self.problem.m1prob)
                + f1.log_prob(torch.Tensor(t1[m1].reshape((t1[m1].shape[0], 1)))).detach().numpy().flatten()
            )
        if m2.sum() > 0:
            llh[m2] = (
                np.log(1 - self.problem.m1prob)
                + f2.log_prob(torch.Tensor(np.column_stack([t1[m2], t2[m2]]))).detach().numpy().flatten()
            )
        return llh

    def getModelIdentifier_old(self):
        """
        Override the built-in method to return the k variable.

        Returns
        -------
        list
            List containing model identifier names

        Notes
        -----
        This is the old version of the method, kept for compatibility.
        TODO: Not used.
        """
        ids = []
        ids.append("k")  # append the nblocks identifier
        for name in self.rv_names:
            i = self.rv[name].getModelIdentifier()
            if i is not None:
                ids.append(i)  # TODO: i or id?
        if len(ids) == 0:
            return None
        else:
            return ids
