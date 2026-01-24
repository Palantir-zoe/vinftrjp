from copy import copy
from typing import Any, ClassVar

import numpy as np
from scipy.special import logsumexp
from scipy.stats import uniform


class Proposal:
    """
    A proposal distribution component for transition kernels in MCMC sampling.

    This class represents a proposal distribution used for generating new states
    in Markov Chain Monte Carlo methods. It can be a single proposal, a layer of
    proposals, or multiple hierarchical layers. Each proposal operates on a vector
    of random variables.

    Notes
    -----
    The current implementation requires integration with ParametricModelSpace
    definitions and proper indexing of random variables.

    Examples
    --------
    >>> ps = Proposal([Componentwise(MVN(), birth(), death())])
    >>> prop_theta = ps.draw(theta, size=N)

    >>> ps2 = Proposal([Componentwise(EigDecComponentwiseNormal(), birth(), death())])
    >>> prop_theta = ps2.draw(theta, size=N)
    """

    idnamedict: ClassVar[dict[int, Any]] = {}
    idpropdict: ClassVar[dict[int, Any]] = {}

    def __init__(self, proposals=[]):
        """
        Initialize Proposal instance.

        Parameters
        ----------
        proposals : list, optional
            List of proposal components or random variable names
        """
        self.ps = []
        self.rv_names = []
        self.rv_indices = {}

        for arg in proposals:
            if isinstance(arg, Proposal):
                self.ps.append(arg)
            elif isinstance(arg, str):
                self.rv_names.append(arg)
            else:
                raise TypeError("Unrecognised type. Should be rv name or proposal.", arg, type(arg))

        print("INIT RV NAMES", self.rv_names, "INIT PROPOSALS", self.ps)

        self.splitby_val = None
        self.setID()
        self.ar = []  # Acceptance rates (0.44 for target default)
        self.t_ar = []  # Timestamps for acceptance rates
        self.n_ar = []  # Number of proposals for each rate
        self.exclude_concat = []

    def make_copy(self):
        """
        Create a deep copy of the proposal instance.

        Returns
        -------
        Proposal
            A new Proposal instance with copied attributes
        """
        to_be_copied = {"rv_names", "rv_indices", "ar", "t_ar", "n_ar"}
        c = copy(self)
        c.__dict__ = {
            attr: copy(self.__dict__[attr]) if attr in to_be_copied else self.__dict__[attr] for attr in self.__dict__
        }
        c.ps = [p.make_copy() for p in self.ps]
        c.setID()
        return c

    def printName(self):
        """Return the class name as string."""
        return self.__class__.__name__

    def setModel(self, m):
        """
        Set the probabilistic model for this proposal.

        Parameters
        ----------
        m : object
            The probabilistic model instance
        """
        self.pmodel = m
        for prop in self.ps:
            prop.setModel(m)

    def getModel(self):
        """Return the associated probabilistic model."""
        return self.pmodel

    @classmethod
    def setAcceptanceRates(cls, prop_id, log_acceptance_ratio, t):
        """
        Update acceptance rates for proposal IDs.

        Parameters
        ----------
        prop_id : array_like
            Array of proposal identifiers
        log_acceptance_ratio : array_like
            Log acceptance ratios for each proposal
        t : int
            Current iteration/timestamp
        """
        unique_pids, pid_indices = np.unique(prop_id, return_inverse=True)
        unique_pids = unique_pids.astype(np.int64)
        for i, pid in enumerate(unique_pids):
            new_ar = np.exp(logsumexp(log_acceptance_ratio[pid_indices == i])) / np.sum(pid_indices == i)
            if not np.isfinite(new_ar):
                print(
                    "Non-finite acceptance rate for prop",
                    pid,
                    "\n",
                    log_acceptance_ratio[pid_indices == i],
                    "\n",
                    np.sum(pid_indices == i),
                )
                new_ar = 0
            cls.idpropdict[int(pid)].setAR(new_ar, t, np.sum(pid_indices == i))

    def setAR(self, cur_ar, t, n):
        """
        Store acceptance rate statistics.

        Parameters
        ----------
        cur_ar : float
            Current acceptance rate
        t : int
            Timestamp/iteration
        n : int
            Number of proposals
        """
        self.ar.append(cur_ar)
        self.t_ar.append(t)
        self.n_ar.append(n)

    def getLastAR(self):
        """Return the most recent acceptance rate."""
        if len(self.ar) > 0:
            return self.ar[-1]
        else:
            return 1.0

    def getAvgARN(self, ll):
        """
        Compute weighted average acceptance rate over recent iterations.

        Parameters
        ----------
        ll : int
            Number of recent iterations to consider

        Returns
        -------
        float
            Weighted average acceptance rate
        """
        if ll > len(self.ar):
            return 0
        return np.dot(np.clip(self.ar[-ll:], None, 1), np.clip(self.n_ar[-ll:], None, 1))

    @classmethod
    def clearIDs(cls):
        """Clear all proposal ID mappings."""
        cls.idnamedict = {}
        cls.idpropdict = {}

    def setID(self):
        """Set unique identifier for this proposal instance."""
        self.idnamedict[id(self)] = self.__class__.__name__
        self.idpropdict[id(self)] = self

    def setModelIdentifier(self, i):
        """
        Set model identifier for this proposal.

        Parameters
        ----------
        i : int
            Model identifier
        """
        self.setID()
        self.model_identifier = i

    def getModelIdentifier(self):
        """Return the model identifier."""
        return self.model_identifier

    def dim(self, model_key):
        """
        Calculate total parameter dimension for a model key.

        Parameters
        ----------
        model_key : hashable
            Identifier for the model

        Returns
        -------
        int
            Total parameter dimension
        """
        ldim = 0
        for name, param_range in self.rv_indices[model_key].items():
            if name not in self.exclude_concat:
                ldim += len(param_range)
        return int(ldim)

    def getIndicesForModelKey(self, theta, model_key):
        """
        Get parameter indices for a specific model key.

        Parameters
        ----------
        theta : ndarray
            Parameter array
        model_key : hashable
            Model identifier

        Returns
        -------
        ndarray
            Indices for the specified model
        """
        model_key_indices, m_tuple_indices = self.getModel().enumerateModels(theta)
        return model_key_indices[model_key]

    def explodeParameters(self, theta, model_key):
        """
        Split parameters by variable name for a specific model.

        Parameters
        ----------
        theta : ndarray
            Parameter array
        model_key : hashable
            Model identifier

        Returns
        -------
        dict
            Dictionary mapping variable names to parameter arrays
        ndarray
            Indices for the model
        """
        self.setID()
        param_dict = {}
        model_key_indices, m_tuple_indices = self.getModel().enumerateModels(theta)
        rows = model_key_indices[model_key].shape[0]
        split_indices = model_key_indices[model_key]

        for name, param_range in self.rv_indices[model_key].items():
            param_dict[name] = theta[split_indices, :][:, param_range]
        return param_dict, split_indices

    def getVariable(self, theta, vname):
        """
        Extract specific variable columns from parameter array.

        Parameters
        ----------
        theta : ndarray
            Parameter array
        vname : str
            Variable name

        Returns
        -------
        ndarray
            Variable values
        """
        columns = self.getModel().generateRVIndices()
        return theta[:, columns[vname]]

    def setVariable(self, theta, vname, values):
        """
        Set values for a specific variable in parameter array.

        Parameters
        ----------
        theta : ndarray
            Parameter array to modify
        vname : str
            Variable name
        values : array_like
            Values to set

        Returns
        -------
        ndarray
            Modified parameter array
        """
        if not isinstance(values, np.ndarray):
            values = np.full(theta.shape[0], values)
        columns = self.getModel().generateRVIndices()
        theta[:, columns[vname]] = values.reshape((values.shape[0], len(columns[vname])))
        return theta

    def concatParameters(self, theta, model_key, return_indices=False):
        """
        Concatenate parameters for homogeneous proposals.

        Parameters
        ----------
        theta : ndarray
            Parameter array
        model_key : hashable
            Model identifier
        return_indices : bool, optional
            Whether to return concatenation indices

        Returns
        -------
        ndarray
            Concatenated parameter matrix
        list, optional
            Concatenation indices if return_indices=True
        """
        self.setID()
        cur_dim = 0
        concat_indices = []
        model_key_indices, m_tuple_indices = self.getModel().enumerateModels(theta)

        try:
            rows = model_key_indices[model_key].shape[0]
        except KeyError:
            available_keys = list(model_key_indices.keys())
            error_msg = (
                f"Model key {model_key} not found in ensemble. "
                f"Available keys: {available_keys}. "
                f"Parameter array shape: {theta.shape}"
            )
            raise KeyError(error_msg) from None

        split_indices = model_key_indices[model_key]
        param_mat = np.zeros((rows, self.dim(model_key)))
        for name, param_range in self.rv_indices[model_key].items():
            if name not in self.exclude_concat:
                thisdim = len(param_range)
                param_mat[:, cur_dim : cur_dim + thisdim] = theta[np.ix_(split_indices, param_range)]
                concat_indices += param_range
                cur_dim += thisdim

        if return_indices:
            return param_mat, concat_indices
        else:
            return param_mat

    def deconcatParameters(self, param_mat, theta, model_key, dest_model_key=None):
        """
        Reverse operation of concatParameters.

        Parameters
        ----------
        param_mat : ndarray
            Concatenated parameter matrix
        theta : ndarray
            Original parameter array
        model_key : hashable
            Source model identifier
        dest_model_key : hashable, optional
            Destination model identifier

        Returns
        -------
        ndarray
            Reconstructed parameter array
        """
        if dest_model_key is None:
            dest_model_key = model_key
        cur_dim = 0
        p_theta = theta.copy()
        model_key_indices, m_tuple_indices = self.getModel().enumerateModels(p_theta)
        rows = model_key_indices[model_key].shape[0]
        split_indices = model_key_indices[model_key]
        for name, param_range in self.rv_indices[dest_model_key].items():
            if name not in self.exclude_concat:
                thisdim = len(param_range)
                p_theta[np.ix_(split_indices, param_range)] = param_mat[:, cur_dim : cur_dim + thisdim]
                cur_dim += thisdim
        return p_theta

    def applyProposedParametersAllModels(self, proposed_dict, theta, model_map=None):
        """
        Apply proposed parameters to all models.

        Parameters
        ----------
        proposed_dict : dict
            Dictionary mapping model keys to proposed parameters
        theta : ndarray
            Original parameter array
        model_map : dict, optional
            Mapping between source and destination model keys

        Returns
        -------
        ndarray
            Updated parameter array
        """
        prop_theta = theta.copy()
        model_key_indices, m_tuple_indices = self.getModel().enumerateModels(theta)
        for model_key, proposed in proposed_dict.items():
            rows = model_key_indices[model_key].shape[0]
            split_indices = model_key_indices[model_key]

            if model_map is not None:
                idx_model_key = model_map[model_key]
            else:
                idx_model_key = model_key
            for name, param_range in self.rv_indices[idx_model_key].items():
                prop_theta[np.ix_(split_indices, param_range)] = proposed[name]
        return prop_theta

    def applyProposedParameters(self, proposed, theta, model_key):
        """
        Apply proposed parameters for a specific model.

        Parameters
        ----------
        proposed : dict
            Proposed parameters dictionary
        theta : ndarray
            Original parameter array
        model_key : hashable
            Model identifier

        Returns
        -------
        ndarray
            Updated parameter array
        """
        prop_theta = theta.copy()
        model_key_indices, m_tuple_indices = self.getModel().enumerateModels(theta)
        rows = model_key_indices[model_key].shape[0]
        split_indices = model_key_indices[model_key]

        for name, param_range in self.rv_indices[model_key].items():
            prop_theta[np.ix_(split_indices, param_range)] = proposed[name]
        return prop_theta

    def initRVIndices(self):
        """Initialize random variable indices (placeholder method)."""
        for k, i in self.rv_indices.items():
            pass
        for prop in self.ps:
            prop.initRVIndices()

    def generateAndMapRVIndices(self, theta, model_keys=None):
        """
        Generate and map random variable indices for all models.

        Parameters
        ----------
        theta : ndarray
            Parameter array
        model_keys : list, optional
            List of model keys to process
        """
        if model_keys is None:
            model_keys = self.getModel().getModelKeys(theta)
        self.initRVIndices()
        for model_key in model_keys:
            proposal_columns = self.getModel().generateRVIndices(model_key=model_key)
            self.mapRVIndices(proposal_columns, model_key=model_key)

    def mapRVIndices(self, rv_index_dict, model_key):
        """
        Map random variable indices for a specific model.

        Parameters
        ----------
        rv_index_dict : dict
            Dictionary of random variable indices
        model_key : hashable
            Model identifier
        """
        if model_key not in self.rv_indices:
            self.rv_indices[model_key] = {}
        self.mapRVIndices_internal(rv_index_dict, model_key=model_key)

        for prop in self.ps:
            prop.mapRVIndices(rv_index_dict, model_key)

    def mapRVIndices_internal(self, rv_index_dict, model_key):
        """
        Internal method for mapping random variable indices.

        Parameters
        ----------
        rv_index_dict : dict
            Dictionary of random variable indices
        model_key : hashable
            Model identifier
        """
        for key, value in rv_index_dict.items():
            if isinstance(value, list):
                if key in self.rv_names:
                    self.rv_indices[model_key][key] = value
            elif isinstance(value, dict):
                if key in self.rv_names:
                    self.rv_indices[model_key][key] = collect_rv_dict_indices(value)
                else:
                    self.mapRVIndices_internal(value, model_key)
            else:
                raise TypeError("Unsupported index type")

    @staticmethod
    def resample_idx(weights, n=None):
        """
        Systematic resampling of indices based on weights.

        Parameters
        ----------
        weights : array_like
            Normalized or unnormalized weights
        n : int, optional
            Number of samples to draw

        Returns
        -------
        ndarray
            Resampled indices
        """
        if n is None:
            n = weights.shape[0]

        indices = np.zeros(n, dtype=np.int32)
        weights = weights / np.sum(weights)
        C = np.cumsum(weights) * n
        u0 = np.random.uniform(0, 1)
        j = 0
        for i in range(n):
            u = u0 + i
            while u > C[j]:
                j += 1
            indices[i] = j
        return indices

    def calibrateweighted(self, theta, weights, m_indices_dict, size, t):
        """
        Calibrate proposal using weighted particles.

        Parameters
        ----------
        theta : ndarray
            Parameter array
        weights : array_like
            Particle weights
        m_indices_dict : dict
            Dictionary mapping model keys to indices
        size : int
            Sample size for calibration
        t : int
            Current iteration/timestamp
        """
        cal = getattr(self, "calibrate", None)
        if callable(cal):
            resampled_theta_list = []
            for i, m_indices in m_indices_dict.items():
                print(f"Theta[{i}].shape = {m_indices.shape[0]}")
                idx = self.resample_idx(weights[m_indices], n=min(2000, m_indices.shape[0]))
                resampled_theta_list.append(theta[m_indices][idx])
            cal(np.vstack(resampled_theta_list), size, t)
        else:
            for prop in self.ps:
                prop.calibrateweighted(theta, weights, m_indices_dict, size, t)

    def calibratemmmpd(self, mmmpd, size, t):
        """
        Calibrate proposal using Multi-Model Mixture Posterior Density.

        Parameters
        ----------
        mmmpd : object
            MultiModelMPD object containing mixture posterior densities
        size : int
            Sample size for calibration
        t : int
            Current iteration/timestamp
        """
        cal = getattr(self, "calibrate", None)
        if callable(cal):
            rs, rs_w = mmmpd.getParticleDensityForTemperature(t, resample=True, resample_max_size=2000)
            cal(rs, size, t)
        else:
            for prop in self.ps:
                prop.calibratemmmpd(mmmpd, size, t)

    def draw(self, theta, size=1):
        """
        Draw samples from proposal distribution (to be implemented by subclasses).

        Parameters
        ----------
        theta : ndarray
            Current parameter values
        size : int, optional
            Number of samples to draw

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses
        """
        raise NotImplementedError


def collect_rv_dict_indices(rv_index_dict):
    rv_indices = []
    for key, value in rv_index_dict.items():
        if isinstance(value, list):
            rv_indices.append(value)
        elif isinstance(value, dict):
            rv_indices.append(collect_rv_dict_indices(value))
    return rv_indices


class RepeatKernel(Proposal):
    """
    Proposal that applies a base kernel multiple times with Metropolis-Hastings.

    Repeatedly applies a base proposal kernel, performing Metropolis-Hastings
    acceptance after each application. This can improve mixing at the cost
    of additional computation.

    Notes
    -----
    - Applies base kernel `nrepeats` times with intermediate acceptance
    - Performs full Metropolis-Hastings step after each application
    - Useful for kernels with low acceptance rates that benefit from multiple attempts
    """

    def __init__(self, kernel, nrepeats=1):
        """
        Initialize repeat kernel.

        Parameters
        ----------
        kernel : Proposal
            Base proposal kernel to repeat
        nrepeats : int, optional
            Number of times to apply the kernel (default: 1)
        """
        super().__init__([kernel])
        self.kernel = kernel
        self.nrepeats = nrepeats

    def calibrate(self, theta, size, t):
        """
        Calibrate the base kernel.

        Parameters
        ----------
        theta : ndarray
            Parameter array for calibration
        size : int
            Sample size for calibration
        t : int
            Current iteration/timestamp
        """
        self.kernel.calibrate(theta, size, t)
        self.t = t

    def draw(self, theta, size=1):
        """
        Apply base kernel multiple times with Metropolis-Hastings acceptance.

        Parameters
        ----------
        theta : ndarray
            Current parameter values
        size : int, optional
            Number of samples to draw (default: 1)

        Returns
        -------
        prop_theta : ndarray
            Proposed parameter values after all repetitions
        prop_lpqratio : ndarray
            Log proposal density ratios
        prop_ids : ndarray
            Proposal identifiers
        """
        N = theta.shape[0]
        # Handle empty input case
        if N == 0:
            return self.kernel.draw(theta, size)

        cur_theta = theta.copy()
        # Precompute initial log-likelihood (inefficient but functional)
        llh = self.pmodel.compute_llh(cur_theta)

        # Apply kernel multiple times with intermediate acceptance
        for i in range(self.nrepeats - 1):
            # Generate proposal from current state
            prop_theta, prop_lpqratio, prop_id = self.kernel.draw(cur_theta, size)
            prop_theta = self.pmodel.sanitise(prop_theta)  # Ensure parameter validity

            # Compute priors for Metropolis-Hastings ratio
            cur_prior = self.pmodel.compute_prior(theta)
            prop_prior = self.pmodel.compute_prior(prop_theta)

            # Identify valid proposals (finite prior and proposal ratio)
            valid_theta = np.logical_and(np.isfinite(prop_prior), np.isfinite(prop_lpqratio))
            prop_llh = np.full(N, np.NINF)  # Initialize with negative infinity

            # Compute likelihood only for valid proposals
            if valid_theta.sum() > 0:
                prop_llh[valid_theta] = self.pmodel.compute_llh(prop_theta[valid_theta, :])

            # Compute Metropolis-Hastings acceptance ratio
            log_acceptance_ratio = self.pmodel.compute_lar(
                cur_theta,
                prop_theta,
                prop_lpqratio,
                prop_llh,
                llh,
                cur_prior,
                prop_prior,
                self.t,
            )

            # Update acceptance rate statistics
            Proposal.setAcceptanceRates(prop_id, log_acceptance_ratio, self.t)

            # Accept/reject decisions
            log_u = np.log(uniform.rvs(0, 1, size=N))
            reject_indices = log_acceptance_ratio < log_u

            # Reject proposals where acceptance ratio < uniform random variable
            prop_theta[reject_indices] = cur_theta[reject_indices]
            cur_theta = prop_theta
            prop_llh[reject_indices] = llh[reject_indices]
            llh = prop_llh

        # Final proposal using the base kernel
        return self.kernel.draw(cur_theta, size)


class UniformChoiceProposal(Proposal):
    """
    Proposal that uniformly randomly selects among multiple sub-proposals.

    Randomly assigns particles to different sub-proposals with equal probability.
    Each particle is processed by exactly one randomly chosen sub-proposal.

    Notes
    -----
    - Uniformly distributes particles among sub-proposals
    - Each sub-proposal handles a subset of particles
    - Useful for combining different proposal strategies
    """

    def draw(self, theta, size=1):
        """
        Randomly assign particles to sub-proposals and generate proposals.

        Parameters
        ----------
        theta : ndarray
            Current parameter values
        size : int, optional
            Number of samples to draw (default: 1)

        Returns
        -------
        prop_theta : ndarray
            Proposed parameter values from all sub-proposals
        prop_lpqratio : ndarray
            Log proposal density ratios
        ids : ndarray
            Proposal identifiers
        """
        n = theta.shape[0]
        prop_theta = np.zeros_like(theta)
        prop_lpqratio = np.zeros(n)

        # Randomly assign each particle to a sub-proposal
        choice = np.random.randint(len(self.ps), size=n)
        ids = np.zeros(n)

        # Process each sub-proposal's assigned particles
        for i in range(len(self.ps)):
            # Get indices of particles assigned to current sub-proposal
            particle_indices = choice == i
            (
                prop_theta[particle_indices],
                prop_lpqratio[particle_indices],
                ids[particle_indices],
            ) = self.ps[i].draw(theta[particle_indices, ...], np.sum(particle_indices))
        return prop_theta, prop_lpqratio, ids


class SystematicChoiceProposal(Proposal):
    """
    Proposal that systematically cycles through multiple sub-proposals.

    This proposal distributes particles systematically among multiple
    sub-proposals, cycling through them in round-robin fashion.
    """

    def __init__(self, proposals=[]):
        """
        Initialize systematic choice proposal.

        Parameters
        ----------
        proposals : list
            List of sub-proposal instances to cycle through
        """
        super().__init__(proposals)
        self.counter = 0  # Current proposal index

    def draw(self, theta, size=1):
        """
        Distribute particles systematically among sub-proposals.

        Particles are divided into approximately equal groups and
        assigned to different sub-proposals in round-robin fashion.

        Parameters
        ----------
        theta : ndarray
            Current parameter values
        size : int, optional
            Number of samples to draw (default: 1)

        Returns
        -------
        prop_theta : ndarray
            Proposed parameter values from all sub-proposals
        prop_lpqratio : ndarray
            Log proposal density ratios
        ids : ndarray
            Proposal identifiers
        """
        n = theta.shape[0]
        prop_theta = np.zeros_like(theta)
        prop_lpqratio = np.zeros(n)
        choice = np.full(n, self.counter)  # Assign all particles to current proposal
        self.counter = (self.counter + 1) % len(self.ps)  # Cycle to next proposal

        ids = np.zeros(n)

        # Process each particle group with its assigned proposal
        for i in range(len(self.ps)):
            (
                prop_theta[choice == i],
                prop_lpqratio[choice == i],
                ids[choice == i],
            ) = self.ps[i].draw(theta[choice == i, ...], np.sum(choice == i))  # TODO: remove size parameter
        return prop_theta, prop_lpqratio, ids


# TODO: Create a new block split proposal that splits by indices from enumerateModels
class ModelEnumerateProposal(Proposal):
    """
    Proposal distribution that manages separate sub-proposals for each model.

    This class maintains independent proposal distributions for different models
    in an ensemble. During calibration, it separates parameters by model using
    enumeration and calibrates each model's proposal separately. During sampling,
    it applies the corresponding proposal to each model's parameters.

    Notes
    -----
    - Uses model enumeration to split parameter arrays by model type
    - Maintains separate proposal instances for each model
    - Handles both calibration and sampling in a model-aware manner
    """

    def __init__(self, subproposal):
        """
        Initialize model-enumerated proposal.

        Parameters
        ----------
        subproposal : Proposal
            Base proposal instance to be replicated for each model
        """
        # Use self.ps[0] for sub-proposal storage
        # splitby_name is the column name to split by (e.g., nlayers or nblocks)
        assert isinstance(subproposal, Proposal)
        super().__init__([subproposal])
        self.blocksplitps = {}  # Dictionary of k-model subproposals

    def enumerateModels(self, theta):
        """
        Enumerate models and separate parameters by model type.

        Parameters
        ----------
        theta : ndarray
            Parameter array containing multiple models

        Returns
        -------
        enumerated_theta_dict : dict
            Dictionary mapping model keys to parameter subsets
        m_indices_dict : dict
            Dictionary mapping model keys to original indices
        """
        m_indices_dict, rev = self.pmodel.enumerateModels(theta)
        enumerated_theta_dict = {}
        for k, idx in m_indices_dict.items():
            enumerated_theta_dict[k] = theta[idx]
        return enumerated_theta_dict, m_indices_dict

    def old_calibrate(self, theta, size, t):
        """
        Calibrate sub-proposals for each model separately.

        Parameters
        ----------
        theta : ndarray
            Parameter array containing multiple models
        size : int
            Sample size for calibration
        t : int
            Current iteration/timestamp
        """
        enumerated_theta_list, m_indices_dict = self.enumerateModels(theta)

        for i, enumerated_theta in enumerated_theta_list.items():
            # Create proposal copy for new model types
            if i not in self.blocksplitps:
                # FIXME: Don't use the below for loop. Use self.ps[0]
                for prop in self.ps:
                    prop2 = prop.make_copy()
                    prop2.setModelIdentifier(i)
                    self.blocksplitps[i] = prop2

            # Calibrate proposal for current model
            self.blocksplitps[i].calibrate(enumerated_theta, size, t)

    def calibratemmmpd(self, mmmpd, size, t):
        """
        Calibrate using Multi-Model Mixture Posterior Density.

        Parameters
        ----------
        mmmpd : object
            MultiModelMPD object containing mixture posterior densities
        size : int
            Sample size for calibration
        t : int
            Current iteration/timestamp
        """
        mklist = self.pmodel.getModelKeys()
        for mk in mklist:
            # Create proposal copy for new model keys
            if mk not in self.blocksplitps:
                # FIXME: Don't use the below for loop. Use self.ps[0]
                for prop in self.ps:
                    prop2 = prop.make_copy()
                    prop2.setModelIdentifier(mk)
                    self.blocksplitps[mk] = prop2
            # Calibrate proposal for current model key
            self.blocksplitps[mk].calibratemmmpd(mmmpd, size, t)

    def draw(self, theta, size=1):
        """
        Draw proposals for each model using corresponding sub-proposal.

        Parameters
        ----------
        theta : ndarray
            Current parameter values across all models
        size : int, optional
            Number of samples to draw (default: 1)

        Returns
        -------
        prop_theta : ndarray
            Proposed parameter values
        prop_lpqratio : ndarray
            Log proposal density ratios
        ids : ndarray
            Proposal identifiers
        """
        n = theta.shape[0]
        prop_theta = np.zeros_like(theta)
        prop_lpqratio = np.zeros(n)
        enumerated_theta_dict, mid = self.enumerateModels(theta)
        ids = np.zeros(n)

        # Apply each model's proposal to its corresponding parameters
        for i, enumerated_theta in enumerated_theta_dict.items():
            mi = mid[i]  # Indices for current model
            # Draw proposals using model-specific proposal distribution
            prop_theta[mi], prop_lpqratio[mi], ids[mi] = self.blocksplitps[i].draw(enumerated_theta, mi.shape[0])
        return prop_theta, prop_lpqratio, ids
