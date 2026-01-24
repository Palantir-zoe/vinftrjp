import itertools

import numpy as np
from matplotlib import pyplot as plt

from src.proposals import Proposal

from .block import RandomVariableBlock

np.set_printoptions(linewidth=200)


class ParametricModelSpace(RandomVariableBlock):
    """
    Parametric model space defining the complete probabilistic model.

    Combines random variables with proposal mechanisms to form a complete
    probabilistic model for MCMC sampling. Supports trans-dimensional models,
    variable transforms, and various proposal calibration methods.

    Examples
    --------
    >>> mymodel = ParametricModelSpace(
    ...     random_variables={
    ...         'ConductiveLayers': TransDimensionalBlock({
    ...             'Conductivity': UniformRV(),
    ...             'Thickness': DirichletRV()
    ...         }),
    ...         'ChargeableLayers': TransDimensionalBlock({
    ...             'Chargeability': UniformRV(),
    ...             'FrequencyDependence': UniformRV(),
    ...             'TimeConstant': UniformRV(),
    ...             'Thickness': DirichletRV()
    ...         }),
    ...         'Geometry1': UniformRV(),
    ...         'Geometry2': UniformRV()
    ...     },
    ...     proposal=[
    ...         EigDecComponentwiseNormalProposal(['ConductiveLayers','ChargeableLayers']),
    ...         BirthDeathProposal(['ConductiveLayers', IndependentProposal('Conductivity', BetaDistribution(alpha, beta)])
    ...     ]
    ... )
    >>> prop_theta = mymodel.propose(theta)

    Notes
    -----
    - Integrates random variables with proposal mechanisms
    - Supports both fixed and trans-dimensional models
    - Provides methods for sampling, density evaluation, and model enumeration
    - Essential component for reversible jump MCMC implementations
    """

    def __init__(self, random_variables, proposal, rv_transforms={}):
        """
        Initialize parametric model space.

        Parameters
        ----------
        random_variables : dict
            Dictionary of random variables and blocks defining the model structure
        proposal : Proposal
            Proposal mechanism for MCMC sampling
        rv_transforms : dict, optional
            Dictionary mapping variable names to transformation functions
        """
        super().__init__(random_variables)
        assert isinstance(proposal, Proposal)
        self.proposal = proposal
        self.proposal.setModel(self)
        self.em_method = "full"  # 'full' or 'block' for estimation of moments
        self.rv_transforms = rv_transforms  # by default, and empty dict.
        # set the model for each random variable so that they can call self.pmodel.getModelIdentifier()
        for rvn in self.rv_names:
            self.rv[rvn].setModel(self)

    def sanitise(self, theta):
        """
        Sanitize parameter values to ensure validity.

        Parameters
        ----------
        theta : ndarray
            Parameter array to sanitize

        Returns
        -------
        ndarray
            Sanitized parameter array
        """
        return theta

    def sampleFromPrior(self, N):
        """
        Generate samples from the prior distribution.

        Parameters
        ----------
        N : int
            Number of samples to generate

        Returns
        -------
        ndarray
            Samples from the prior distribution
        """
        # sample from prior
        theta = np.zeros((N, self.dim()))
        cur_dim = 0
        for k in self.rv_names:
            thisdim = self.rv[k].dim()
            theta[:, cur_dim : cur_dim + thisdim] = self.rv[k].draw(N).reshape(N, thisdim)
            cur_dim += thisdim
        return theta

    def assertDimension(self, theta):
        """
        Validate parameter array dimensions.

        Parameters
        ----------
        theta : ndarray
            Parameter array to validate

        Raises
        ------
        AssertionError
            If parameter dimensions don't match model dimensions
        """
        assert theta.shape[1] == self.dim(), (
            "Theta n cols {} is not equal to dimension of ParametricModelSpace {}".format(theta.shape[1], self.dim())
        )

    def dim(self, model_key=None):
        """
        Get model dimension, optionally conditioned on model key.

        Parameters
        ----------
        model_key : hashable, optional
            Model key for conditional dimension calculation

        Returns
        -------
        int
            Model dimension
        """
        if model_key is None:
            return super().dim()
        else:
            ldim = 0
            for k in self.rv_names:
                ldim += self.rv[k].dim(model_key)
            return int(ldim)

    @staticmethod
    def plotJoints(theta, prop_theta, propname):
        """
        Plot joint distributions of current and proposed parameters.

        Parameters
        ----------
        theta : ndarray
            Current parameter samples
        prop_theta : ndarray
            Proposed parameter samples
        propname : str
            Name of proposal for plot title
        """
        nlayers, nlidx = np.unique(theta[:, 0].astype(int), return_inverse=True)
        for k in nlayers:
            ncols = int(k * 2 + 1)
            nrows = ncols
            thetaidx = theta[:, 0] == k
            propidx = prop_theta[:, 0] == k
            fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True)
            fig.suptitle(propname)
            colidx = (
                list(range(1, k + 1))
                + list(range(self.max_layers + 1, self.max_layers + 1 + k))
                + [self.max_layers * 2 + 1]
            )
            paramtheta = theta[thetaidx, :][:, colidx]
            paramprop = prop_theta[propidx, :][:, colidx]
            for i in range(2 * k + 1):
                for j in range(2 * k + 1):
                    if k == 0:
                        thisaxs = axs
                    else:
                        thisaxs = axs[i, j]
                    if i == j:
                        n, bins, patches = thisaxs.hist(paramtheta[:, i], color="blue", density=True, bins=20)
                        thisaxs.hist(paramprop[:, i], density=True, color="red", rwidth=0.5, bins=20)
                        if k > 0:
                            for p in range(i, 2 * k + 1):
                                thisaxs.get_shared_y_axes().remove(axs[i, p])
                    elif i > j:
                        thisaxs.scatter(paramtheta[:, j], paramtheta[:, i], color="blue", s=0.2)
                        thisaxs.scatter(paramprop[:, j], paramprop[:, i], color="red", s=0.2)
                        thisaxs.get_shared_y_axes().remove(axs[j, j])
                    else:
                        fig.delaxes(thisaxs)
                    if i < 2 * k and k != 0:
                        thisaxs.xaxis.set_ticks_position("none")
                    if j > 0:
                        thisaxs.yaxis.set_ticks_position("none")

            plt.show()

    def getModelKeysFromSpec(self):
        """
        Generate all possible model keys from model specification.

        Returns
        -------
        list
            List of all possible model key tuples
        """
        ids = self.getModelIdentifier()
        # for each rv name, get the range
        if ids is not None:
            rvrangelist = []
            for i in ids:
                if i is not None:
                    rv = self.retrieveRV(i)
                    if rv is not None:
                        rvrangelist.append(rv.getRange())
                    else:
                        raise ValueError("Key error, rv name {i} not found in model")
            # get all permutations
            keylist = []
            for p in itertools.product(*rvrangelist):
                keylist.append(tuple(p))
            return keylist
        else:
            return ()  # empty identifier

    def getModelKeys(self, theta=None):
        """
        Get model keys from specification or parameter array.

        Parameters
        ----------
        theta : ndarray, optional
            Parameter array to extract model keys from

        Returns
        -------
        list
            List of model key tuples
        """
        if theta is None:
            return self.getModelKeysFromSpec()
        ids = self.getModelKeyColumns()
        tags = theta[:, ids]
        unique_rows, tuple_indices = np.unique(tags, return_inverse=True, axis=0)
        # convert each unique row to immutable tuple for use as a dict key.
        # return list of these tuples and a numpy array of indices to this list
        return list(map(tuple, unique_rows))

    def getModelKeyColumns(self):
        """
        Get column indices for model key variables.

        Returns
        -------
        list
            List of column indices for model key variables
        """
        ids = self.getModelIdentifier()
        rv_index_dict = (
            self.generateRVIndices()
        )  # DO NOT PASS MODEL KEY: generateRVIndices() calls this getModelKeyColumns() method to obtain keys.
        indices = []
        for k in ids:
            if k is not None:
                indices += rv_index_dict[k]
        return indices

    def enumerateModels(self, theta):
        """
        Enumerate models and group parameters by model type.

        Parameters
        ----------
        theta : ndarray
            Parameter array to enumerate

        Returns
        -------
        model_enumeration : dict
            Dictionary mapping model keys to parameter indices
        tuple_indices : ndarray
            Array mapping each parameter to its model key index
        """
        indices = self.getModelKeyColumns()
        # use to index theta
        tags = theta[:, indices]
        unique_rows, tuple_indices = np.unique(tags, return_inverse=True, axis=0)
        # convert each unique row to immutable tuple for use as a dict key.
        # return list of these tuples and a numpy array of indices to this list
        keys = list(map(tuple, unique_rows))
        model_enumeration = {}
        for i, k in enumerate(keys):
            model_enumeration[k] = np.where(tuple_indices == i)[0]
        return model_enumeration, tuple_indices

    def hasConditionalApproximation(self, mk):
        """
        Check if conditional MVN approximation is available for model key.

        Parameters
        ----------
        mk : hashable
            Model key to check

        Returns
        -------
        bool
            True if conditional approximation is available
        """
        return False

    def getConditionalApproximation(self, mk):
        """
        Get conditional MVN approximation for model key.

        Parameters
        ----------
        mk : hashable
            Model key

        Returns
        -------
        object
            Conditional approximation object
        """
        pass

    def useBarycenterCombination(self):
        """
        Check if barycenter combination should be used.

        Returns
        -------
        bool
            True if barycenter combination should be used
        """
        return False

    def calibrateProposalsWeighted(self, theta, weights, N, t):
        """
        Calibrate proposals using weighted particles.

        Parameters
        ----------
        theta : ndarray
            Parameter samples
        weights : ndarray
            Particle weights
        N : int
            Sample size for calibration
        t : int
            Current iteration/timestamp
        """
        # calibrate the proposals
        self.proposal.generateAndMapRVIndices(theta)
        m_indices, rev = self.enumerateModels(theta)
        self.proposal.calibrateweighted(theta, weights, m_indices, N, t)

    def calibrateProposalsMMMPD(self, mmmpd, N, t):
        """
        Calibrate proposals using Multi-Model Mixture Posterior Density.

        Parameters
        ----------
        mmmpd : object
            MultiModelMPD object
        N : int
            Sample size for calibration
        t : int
            Current iteration/timestamp
        """
        # calibrate the proposals
        self.proposal.generateAndMapRVIndices(theta=None, model_keys=mmmpd.getModelKeys())
        self.proposal.calibratemmmpd(mmmpd, N, t)

    def calibrateProposalsUnweighted(self, theta, N, t):
        """
        Calibrate proposals using unweighted particles.

        Parameters
        ----------
        theta : ndarray
            Parameter samples
        N : int
            Sample size for calibration
        t : int
            Current iteration/timestamp
        """
        # calibrate the proposals
        self.proposal.generateAndMapRVIndices(theta)
        N2 = theta.shape[0]
        m_indices, rev = self.enumerateModels(theta)
        self.proposal.calibrateweighted(theta, np.full(N2, 1.0 / N2), m_indices, N, t)

    def deconcatAllParameters(self, param_mat, theta, model_key, dest_model_key=None, transform=False):
        """
        Reverse parameter concatenation operation.

        Parameters
        ----------
        param_mat : ndarray
            Concatenated parameter matrix
        theta : ndarray
            Original parameter array
        model_key : hashable
            Source model key
        dest_model_key : hashable, optional
            Destination model key
        transform : bool, optional
            Whether to apply transforms (not implemented)

        Returns
        -------
        ndarray
            Deconcatenated parameter array
        """
        self.proposal.generateAndMapRVIndices(theta, model_keys=self.getModelKeys())
        if transform:
            raise Exception("deconcatAllParameters() not implemented with transforms")
        else:
            return self.proposal.deconcatParameters(param_mat, theta, model_key, dest_model_key)

    def concatAllParameters(self, theta, mk, transform=False):
        """
        Concatenate parameters for homogeneous processing.

        Parameters
        ----------
        theta : ndarray
            Parameter array
        mk : hashable
            Model key
        transform : bool, optional
            Whether to apply variable transforms

        Returns
        -------
        ndarray
            Concatenated parameter matrix
        """
        self.proposal.generateAndMapRVIndices(theta, model_keys=self.getModelKeys())
        if transform:
            # use self.rv_transforms dict to transform each parameter
            params, splitidx = self.proposal.explodeParameters(theta, mk)
            for rv_name, tf in self.rv_transforms.items():
                params[rv_name] = tf(params[rv_name])
            Ttheta = self.proposal.applyProposedParameters(params, theta, model_key=mk)
            return self.proposal.concatParameters(Ttheta, mk)
        else:
            return self.proposal.concatParameters(theta, mk)

    def _estimatemoments(self, theta, mk):
        """
        Estimate moments for parameters of specific model type.

        Parameters
        ----------
        theta : ndarray
            Parameter samples
        mk : hashable
            Model key

        Returns
        -------
        mean : ndarray
            Mean vector
        cov : ndarray
            Covariance matrix
        """
        if self.em_method == "block":
            rv_dim_dict = self.generateRVIndices(
                flatten_tree=False
            )  # do not pass model key, leave that to TransDimensionalBlock et al
            ncols = int(self.dim(mk))
            mean = np.zeros(ncols)
            cov = np.eye(ncols)  # by default, identity.
            curdim = 0
            # for key,rvdim in rv_dim_dict.items():
            for key in self.rv_names:
                rvdim = self.rv[key].dim(mk)
                print(
                    "ParametricModelSpace: Estimating moments for ",
                    key,
                    " dim ",
                    rvdim,
                    " idx ",
                    rv_dim_dict[key],
                )
                (
                    mean[curdim : curdim + rvdim],
                    cov[curdim : curdim + rvdim, curdim : curdim + rvdim],
                ) = self.rv[key]._estimatemoments(theta[:, rv_dim_dict[key]], mk)
                curdim += rvdim
            return mean, cov
        elif self.em_method == "full":
            # Assume all rows are for model mk
            theta_k = self.concatAllParameters(theta, mk, transform=True)
            # print("theta_k",theta_k)
            mean = np.mean(theta_k, axis=0)
            cov = np.cov(theta_k.T)
            return mean, cov
        else:
            raise Exception("Unsupported moment estimation method: {}".format(self.em_method))

    def propose(self, theta, N):
        """
        Generate proposals using the configured proposal mechanism.

        Parameters
        ----------
        theta : ndarray
            Current parameter values
        N : int
            Number of proposals to generate

        Returns
        -------
        ndarray
            Proposed parameter values
        """
        self.assertDimension(theta)
        # call the top level proposal.
        self.proposal.generateAndMapRVIndices(theta)
        # subset theta to only columns being proposed
        return self.proposal.draw(theta, N)

    def compute_prior(self, theta):
        """
        Compute log prior density for parameters.

        Parameters
        ----------
        theta : ndarray
            Parameter values

        Returns
        -------
        ndarray
            Log prior density values
        """
        # traverse RVs and compute prior for each.
        # returns log of the prior evaluated at theta
        n = theta.shape[0]
        rv_dim_dict = self.generateRVIndices(flatten_tree=False)
        prop_prior = np.zeros(n)
        for key in self.rv_names:
            prop_prior += self.rv[key].eval_log_prior(theta[:, rv_dim_dict[key]]).reshape(n)
        return prop_prior

    def compute_llh(self, theta):
        """
        Compute log likelihood for parameters.

        Parameters
        ----------
        theta : ndarray
            Parameter values

        Returns
        -------
        ndarray
            Log likelihood values
        """
        return 1

    def setStartingDistribution(self, starting_dist):
        """
        Set starting distribution for annealing.

        Parameters
        ----------
        starting_dist : object
            Starting distribution object
        """
        # TODO assert is correct distribution
        self.starting_dist = starting_dist

    def evalStartingDistribution(self, theta):
        """
        Evaluate starting distribution density.

        Parameters
        ----------
        theta : ndarray
            Parameter values

        Returns
        -------
        ndarray
            Starting distribution density values
        """
        return self.starting_dist.compute_prior(theta)

    def compute_lar(
        self,
        cur_theta,
        prop_theta,
        prop_lpqratio,
        prop_llh,
        llh,
        cur_prior,
        prop_prior,
        temperature,
    ):
        """
        Compute log acceptance ratio for Metropolis-Hastings.

        Parameters
        ----------
        cur_theta : ndarray
            Current parameter values
        prop_theta : ndarray
            Proposed parameter values
        prop_lpqratio : ndarray
            Log proposal density ratio
        prop_llh : ndarray
            Proposed log likelihood
        llh : ndarray
            Current log likelihood
        cur_prior : ndarray
            Current log prior
        prop_prior : ndarray
            Proposed log prior
        temperature : float
            Annealing temperature

        Returns
        -------
        ndarray
            Log acceptance ratios
        """
        cur_start_dist = self.evalStartingDistribution(cur_theta)
        prop_start_dist = self.evalStartingDistribution(prop_theta)
        lar = (
            temperature * (prop_llh - llh + prop_prior - cur_prior)
            + (1 - temperature) * (prop_start_dist - cur_start_dist)
            + prop_lpqratio
        )
        lar[np.isnan(lar)] = -np.inf
        lar[np.isneginf(lar)] = -np.inf
        lar[np.isposinf(lar)] = 0
        lar[lar > 0] = 0
        return lar
