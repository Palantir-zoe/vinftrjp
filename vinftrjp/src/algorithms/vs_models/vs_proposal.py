import sys

import numpy as np
import torch
from nflows.distributions.normal import StandardNormal
from scipy.special import logsumexp
from scipy.stats import norm

from src.flows import Flow
from src.proposals import Proposal
from src.transforms import IdentityTransform, Sigmoid


class RJZGlobalBlockVSProposalIndiv(Proposal):
    """
    Individual reversible-jump proposal with per-model normalizing flows.

    Implements trans-dimensional MCMC with separate flow-based proposals
    for each model. Handles variable selection using gamma indicators and
    beta coefficients with model-specific transformations.
    """

    def __init__(
        self,
        problem,
        *,
        blocksizes,
        blocknames,
        gammanames,
        betanames,
        within_model_proposal,
        propose_all=False,
        use_opr_calib=False,
        **kwargs,
    ):
        """
        Initialize individual reversible-jump proposal.

        Parameters
        ----------
        blocksizes : list
            Sizes of parameter blocks
        blocknames : list
            Names of parameter blocks
        gammanames : list
            Names of binary indicator variables (gamma)
        betanames : list
            Names of coefficient variables (beta)
        within_model_proposal : Proposal
            Proposal for within-model updates
        propose_all : bool, optional
            Whether to fit proposals for models with zero particles (default: False)
        use_opr_calib : bool, optional
            Whether to use OPR calibration method (default: False)
        """
        self.problem = problem

        self.propose_all = propose_all  # Fit proposals even for models with zero particles
        self.blocksizes = blocksizes
        self.blocknames = blocknames
        self.gammanames = gammanames
        self.betanames = betanames
        assert isinstance(within_model_proposal, Proposal)

        self.within_model_proposal = within_model_proposal
        self.rv_names = gammanames + betanames
        super().__init__([*self.rv_names, self.within_model_proposal])

        self.exclude_concat = gammanames  # Exclude gamma from concatenation
        self.use_opr_calib = use_opr_calib

        self.save_flows_dir = kwargs.get("save_flows_dir", "")

    def calibratemmmpd(self, mmmpd, size, t):
        """
        Calibrate proposal using Multi-Model Mixture Posterior Density.

        Estimates model evidence and trains individual normalizing flows
        for each model in the ensemble.

        Parameters
        ----------
        mmmpd : object
            MultiModelMPD object containing mixture posterior densities
        size : int
            Sample size for calibration
        t : int
            Current iteration/timestamp
        """
        # Calibrate within-model proposal
        self.within_model_proposal.calibratemmmpd(mmmpd, size, t)
        mklist = self.pmodel.getModelKeys()  # Get all model keys
        cols = self.pmodel.generateRVIndices()

        self.flows = {}  # Dictionary to store flows for each model
        self.mk_logZhat = {}  # Dictionary to store model evidence estimates

        # Estimate model evidence using different calibration methods
        if self.use_opr_calib:
            # Use OPR calibration method
            full_theta, full_theta_w = mmmpd.getParticleDensityForTemperature(t, resample=False)
            full_mkdict, rev = self.pmodel.enumerateModels(full_theta)
            full_N = full_theta.shape[0]

            for mk in mklist:
                if mk in full_mkdict.keys():
                    # Estimate log evidence from weighted particles
                    self.mk_logZhat[mk] = logsumexp(full_theta_w[full_mkdict[mk]]) - np.log(full_N)
                else:
                    self.mk_logZhat[mk] = -np.inf  # No particles for this model
        else:
            # Use original calibration method
            orig_theta, orig_theta_w = mmmpd.getOriginalParticleDensityForTemperature(
                t, resample=True, resample_max_size=10000
            )
            orig_mkdict, rev = self.pmodel.enumerateModels(orig_theta)

            for mk in mklist:
                if mk in orig_mkdict.keys():
                    # Estimate log evidence from particle counts
                    self.mk_logZhat[mk] = np.log(orig_mkdict[mk].shape[0] * 1.0 / 10000)
                else:
                    self.mk_logZhat[mk] = -np.inf  # No particles for this model

        # Compute model transition probabilities
        pp_mk_logZ = np.zeros(len(self.mk_logZhat.keys()))
        pp_mk_keys = []
        for i, (pmk, pmk_logZ) in enumerate(self.mk_logZhat.items()):
            pp_mk_logZ[i] = pmk_logZ
            pp_mk_keys.append(pmk)
        pp_mk_log_prob = pp_mk_logZ - logsumexp(pp_mk_logZ)
        pp_mk_prob = np.exp(pp_mk_log_prob)

        # Train individual flows for each model
        orig_theta, orig_theta_w = mmmpd.getOriginalParticleDensityForTemperature(
            t, resample=True, resample_max_size=10000
        )
        orig_mkdict, rev = self.pmodel.enumerateModels(orig_theta)

        for mk in mklist:
            if self.use_opr_calib:
                # Use OPR method to get model-specific particles
                mk_theta, mk_theta_w = mmmpd.getParticleDensityForModelAndTemperature(mk, t, resample=False)
                self.flows[mk] = self.makeFlow(
                    mk,
                    mk_theta,
                    np.exp(np.log(mk_theta_w) - logsumexp(np.log(mk_theta_w))),
                )
            else:
                # Use original calibration particles
                if mk in orig_mkdict.keys():
                    mk_theta = orig_theta[orig_mkdict[mk]]
                    mk_theta_w = orig_theta_w[orig_mkdict[mk]]

                    self.flows[mk] = self.makeFlow(
                        mk,
                        mk_theta,
                        np.exp(np.log(mk_theta_w) - logsumexp(np.log(mk_theta_w))),
                    )
                else:
                    # Create dummy flow for models without particles
                    self.flows[mk] = self.makeDummyFlow(mk)

    def makeDummyFlow(self, mk):
        """
        Create identity flow for models without training data.

        Parameters
        ----------
        mk : tuple
            Model key

        Returns
        -------
        Flow
            Identity flow with standard normal base distribution
        """
        dim = self.getModelDim(mk)
        bnorm = StandardNormal((dim,))
        fn = IdentityTransform()
        return Flow(fn, bnorm)

    def makeFlow(self, mk, mk_theta, mk_theta_w):
        """
        Create normalizing flow for a specific model.

        Parameters
        ----------
        mk : tuple
            Model key
        mk_theta : ndarray
            Model parameters for training
        mk_theta_w : ndarray
            Parameter weights for training

        Returns
        -------
        Flow
            Trained normalizing flow for the model
        """
        ls = Sigmoid()
        X = self.concatParameters(mk_theta, mk)
        beta_dim = X.shape[1]
        bnorm = StandardNormal((beta_dim,))

        # Check for numerical stability
        if ~np.any(np.isfinite(np.std(X, axis=0))):
            print(X)
            print("X is singular", X)
            sys.exit(0)

        return self._make_flow(mk, mk_theta_w, ls, X, bnorm)

    def _make_flow(self, mk, mk_theta_w, ls, X, bnorm):
        raise NotImplementedError

    def transformToBase(self, inputs, mk):
        """
        Transform parameters to base distribution using model-specific flow.

        Parameters
        ----------
        inputs : ndarray
            Input parameters
        mk : tuple
            Model key

        Returns
        -------
        transformed : ndarray
            Parameters in base distribution space
        logdet : ndarray
            Log determinant of transformation Jacobian
        """
        if self.getModelDim(mk) == 0:
            return inputs, np.zeros(inputs.shape[0])
        X = self.concatParameters(inputs, mk)
        XX, logdet = self.flows[mk]._transform.forward(torch.tensor(X, dtype=torch.float32))
        return (
            self.deconcatParameters(XX.detach().numpy(), inputs, mk),
            logdet.detach().numpy(),
        )

    def transformFromBase(self, inputs, mk):
        """
        Transform parameters from base distribution back to original space.

        Parameters
        ----------
        inputs : ndarray
            Parameters in base distribution space
        mk : tuple
            Model key

        Returns
        -------
        transformed : ndarray
            Parameters in original space
        logdet : ndarray
            Log determinant of transformation Jacobian
        """
        if self.getModelDim(mk) == 0:
            return inputs, np.zeros(inputs.shape[0])
        X = self.concatParameters(inputs, mk)
        XX, logdet = self.flows[mk]._transform.inverse(torch.tensor(X, dtype=torch.float32))
        return (
            self.deconcatParameters(XX.detach().numpy(), inputs, mk),
            logdet.detach().numpy(),
        )

    def getModelDim(self, mk):
        """
        Calculate parameter dimension for a model.

        Parameters
        ----------
        mk : tuple
            Model key (gamma vector)

        Returns
        -------
        int
            Total parameter dimension for active blocks
        """
        return int(np.array([bs * list(mk)[i] for i, bs in enumerate(self.blocksizes)]).sum())

    def toggleGamma(self, mk, tidx):
        """
        Toggle gamma indicator at specified index.

        Parameters
        ----------
        mk : tuple
            Current model key (gamma vector)
        tidx : int
            Index to toggle

        Returns
        -------
        tuple
            New model key with toggled gamma
        """
        mkl = list(mk)
        mkl[tidx] = 1 - mkl[tidx]
        return tuple(mkl)

    def toggleIDX(self, mk, new_mk):
        """
        Find indices where gamma indicators change between models.

        Parameters
        ----------
        mk : tuple
            Source model key
        new_mk : tuple
            Target model key

        Returns
        -------
        on_indices : ndarray
            Indices where gamma turns on (0→1)
        off_indices : ndarray
            Indices where gamma turns off (1→0)
        """
        mkl = np.array(list(mk))
        new_mkl = np.array(list(new_mk))
        on = np.logical_and(new_mkl, np.logical_not(mkl))
        off = np.logical_and(mkl, np.logical_not(new_mkl))
        return np.where(on)[0], np.where(off)[0]

    def draw(self, theta, size=1):
        """
        Generate proposals with between-model and within-model moves.

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
        logpqratio : ndarray
            Log proposal density ratios
        prop_ids : ndarray
            Proposal identifiers
        """
        logpqratio = np.zeros(theta.shape[0])
        proposed = {}
        mk_idx = {}
        lpq_mk = {}
        prop_ids = np.zeros(theta.shape[0])
        sigma_u = 1  # Standard deviation of auxiliary distribution

        prop_theta = theta.copy()

        # Compute model transition probabilities
        pp_mk_logZ = np.zeros(len(self.mk_logZhat.keys()))
        pp_mk_keys = []
        for i, (pmk, pmk_logZ) in enumerate(self.mk_logZhat.items()):
            pp_mk_logZ[i] = pmk_logZ
            pp_mk_keys.append(pmk)
        pp_mk_log_prob = pp_mk_logZ - logsumexp(pp_mk_logZ)

        # Process each model separately
        model_enumeration, rev = self.pmodel.enumerateModels(theta)
        for mk, mk_row_idx in model_enumeration.items():
            mk_theta = theta[mk_row_idx]
            mk_n = mk_theta.shape[0]
            params, splitdict = self.explodeParameters(theta, mk)
            betas = np.column_stack([params[xn] for xn in self.betanames])  # Active coefficients
            gammas = np.column_stack([params[xn] for xn in self.gammanames])  # All gamma indicators
            gamma_vec = gammas[0]  # Gamma vector (identical across particles)

            # Transform to base distribution
            Tmktheta, lpq1 = self.transformToBase(mk_theta, mk)

            # Compute model transition probabilities
            this_log_prob = self.mk_logZhat[mk] - logsumexp(pp_mk_logZ)
            pidx = np.random.choice(np.arange(pp_mk_logZ.shape[0]), p=np.exp(pp_mk_log_prob), size=mk_n)
            lpq_mk[mk] = this_log_prob - pp_mk_log_prob[pidx]

            proposed[mk], mk_idx[mk] = self.explodeParameters(Tmktheta, mk)

            prop_mk_theta = mk_theta.copy()
            mk_prop_ids = np.zeros(mk_n)

            # Apply proposals based on target model
            for p_i in np.unique(pidx):
                tn = (pidx == p_i).sum()
                new_mk = pp_mk_keys[p_i]
                at_idx = pidx == p_i
                if new_mk == mk:
                    # Within-model update
                    (
                        prop_mk_theta[at_idx],
                        lpq_mk[mk][at_idx],
                        mk_prop_ids[at_idx],
                    ) = self.within_model_proposal.draw(mk_theta[at_idx], tn)
                    continue

                # get column ids for toggled on and off blocks
                on_idx, off_idx = self.toggleIDX(mk, new_mk)
                # toggle ons
                for idx in on_idx:
                    for i in range(self.blocksizes[idx]):
                        cbs = int(np.array(self.blocksizes[:idx]).sum())
                        u = norm(0, sigma_u).rvs(tn)
                        log_u = norm(0, sigma_u).logpdf(u)  # for naive, mean is u
                        lpq_mk[mk][at_idx] -= log_u
                        Tmktheta[at_idx] = self.setVariable(
                            Tmktheta[at_idx], self.betanames[cbs + i], u
                        )  # for naive, update with random walk using old u
                    Tmktheta[at_idx] = self.setVariable(Tmktheta[at_idx], self.gammanames[idx], np.ones(tn))
                # for toggle off
                for idx in off_idx:
                    for i in range(self.blocksizes[idx]):
                        cbs = int(np.array(self.blocksizes[:idx]).sum())
                        u = self.getVariable(Tmktheta[at_idx], self.betanames[cbs + i]).flatten()
                        log_u = norm(0, sigma_u).logpdf(u)  # for naive, mean is u
                        lpq_mk[mk][at_idx] += log_u
                    Tmktheta[at_idx] = self.setVariable(Tmktheta[at_idx], self.gammanames[idx], np.zeros(tn))
                # transform back
                prop_mk_theta[at_idx], lpq2 = self.transformFromBase(Tmktheta[at_idx], new_mk)
                lpq_mk[mk][at_idx] += lpq1[at_idx] + lpq2
                mk_prop_ids[at_idx] = np.full(tn, id(self))

            prop_theta[mk_row_idx] = prop_mk_theta
            logpqratio[mk_row_idx] = lpq_mk[mk]
            prop_ids[mk_row_idx] = mk_prop_ids

        return prop_theta, logpqratio, prop_ids


class RJZGlobalRobustBlockVSProposalSaturated(Proposal):
    """
    Robust reversible-jump proposal with variable selection and flow-based transformations.

    Implements a trans-dimensional MCMC proposal that handles variable selection
    using gamma indicators and beta coefficients. Uses normalizing flows for
    efficient between-model transitions and within-model updates.
    """

    def __init__(self, problem, *, blocksizes, blocknames, gammanames, betanames, within_model_proposal, **kwargs):
        """
        Initialize robust reversible-jump proposal.

        Parameters
        ----------
        blocksizes : list
            Sizes of parameter blocks
        blocknames : list
            Names of parameter blocks
        gammanames : list
            Names of binary indicator variables (gamma)
        betanames : list
            Names of coefficient variables (beta)
        within_model_proposal : Proposal
            Proposal for within-model updates
        """
        self.problem = problem

        self.blocksizes = blocksizes
        self.blocknames = blocknames
        self.gammanames = gammanames
        self.betanames = betanames
        assert isinstance(within_model_proposal, Proposal)
        self.within_model_proposal = within_model_proposal
        self.rv_names = gammanames + betanames

        super().__init__([*self.rv_names, self.within_model_proposal])
        self.exclude_concat = gammanames  # Exclude gamma from concatenation

        self.save_flows_dir = kwargs.get("save_flows_dir", "")

    def calibratemmmpd(self, mmmpd, size, t):
        """
        Calibrate proposal using Multi-Model Mixture Posterior Density.

        Estimates model evidence, trains normalizing flows, and sets up
        between-model transition probabilities.

        Parameters
        ----------
        mmmpd : object
            MultiModelMPD object containing mixture posterior densities
        size : int
            Sample size for calibration
        t : int
            Current iteration/timestamp
        """
        # Calibrate within-model proposal
        self.within_model_proposal.calibratemmmpd(mmmpd, size, t)

        # Estimate model evidence (log Z) for each model
        mklist = self.pmodel.getModelKeys()
        cols = self.pmodel.generateRVIndices()
        orig_theta, orig_theta_w = mmmpd.getOriginalParticleDensityForTemperature(
            t, resample=True, resample_max_size=10000
        )
        orig_mkdict, rev = self.pmodel.enumerateModels(orig_theta)

        self.mk_logZhat = {}
        for mk in mklist:
            if mk in orig_mkdict.keys():
                # Estimate log evidence from particle counts
                self.mk_logZhat[mk] = np.log(orig_mkdict[mk].shape[0] * 1.0 / 10000)
            else:
                self.mk_logZhat[mk] = -np.inf  # No particles for this model

        self.flow = self._make_flow(mklist, mmmpd, t)

        # Compute between-model transition probabilities
        pp_mk_logZ = np.zeros(len(self.mk_logZhat.keys()))
        pp_mk_keys = []
        for i, (pmk, pmk_logZ) in enumerate(self.mk_logZhat.items()):
            pp_mk_logZ[i] = pmk_logZ
            pp_mk_keys.append(pmk)
        pp_mk_log_prob = pp_mk_logZ - logsumexp(pp_mk_logZ)
        pp_mk_prob = np.exp(pp_mk_log_prob)

        def anneal_down(a, logp, thres):
            """Helper function for probability annealing (currently unused)."""
            alogp = a * logp
            npa = np.exp(alogp - logsumexp(alogp))
            return np.max(thres - npa)

    def _make_flow(self, mklist, mmmpd, t):
        raise NotImplementedError

    def transformToBase(self, inputs, mk):
        """
        Transform parameters to base distribution using trained flow.

        Parameters
        ----------
        inputs : ndarray
            Input parameters
        mk : tuple
            Model key

        Returns
        -------
        transformed : ndarray
            Parameters in base distribution space
        logdet : ndarray
            Log determinant of transformation Jacobian
        """
        X = self.concatParameters(inputs, mk)
        Y = np.tile(np.array(list(mk), dtype=np.float32), (X.shape[0], 1))
        XX, logdet = self.flow._transform.forward(
            torch.tensor(X, dtype=torch.float32),
            context=torch.tensor(Y, dtype=torch.float32),
        )
        # Apply volume-preserving permutation
        XXn = XX.detach().numpy()
        XXn = XXn[np.arange(len(XXn))[:, None], np.random.randn(*XXn.shape).argsort(axis=1)]
        return self.deconcatParameters(XXn, inputs, mk), logdet.detach().numpy()

    def transformFromBase(self, inputs, mk):
        """
        Transform parameters from base distribution back to original space.

        Parameters
        ----------
        inputs : ndarray
            Parameters in base distribution space
        mk : tuple
            Model key

        Returns
        -------
        transformed : ndarray
            Parameters in original space
        logdet : ndarray
            Log determinant of transformation Jacobian
        """
        X = self.concatParameters(inputs, mk)
        Y = np.tile(np.array(list(mk), dtype=np.float32), (X.shape[0], 1))
        XX, logdet = self.flow._transform.inverse(
            torch.tensor(X, dtype=torch.float32),
            context=torch.tensor(Y, dtype=torch.float32),
        )
        return (
            self.deconcatParameters(XX.detach().numpy(), inputs, mk),
            logdet.detach().numpy(),
        )

    def toggleGamma(self, mk, tidx):
        """
        Toggle gamma indicator at specified index.

        Parameters
        ----------
        mk : tuple
            Current model key (gamma vector)
        tidx : int
            Index to toggle

        Returns
        -------
        tuple
            New model key with toggled gamma
        """
        mkl = list(mk)
        mkl[tidx] = 1 - mkl[tidx]
        return tuple(mkl)

    def toggleIDX(self, mk, new_mk):
        """
        Find indices where gamma indicators change between models.

        Parameters
        ----------
        mk : tuple
            Source model key
        new_mk : tuple
            Target model key

        Returns
        -------
        on_indices : ndarray
            Indices where gamma turns on (0→1)
        off_indices : ndarray
            Indices where gamma turns off (1→0)
        """
        mkl = np.array(list(mk))
        new_mkl = np.array(list(new_mk))
        on = np.logical_and(new_mkl, np.logical_not(mkl))
        off = np.logical_and(mkl, np.logical_not(new_mkl))
        return np.where(on)[0], np.where(off)[0]

    def draw(self, theta, size=1):
        """
        Generate proposals with between-model and within-model moves.

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
        logpqratio : ndarray
            Log proposal density ratios
        prop_ids : ndarray
            Proposal identifiers
        """
        logpqratio = np.zeros(theta.shape[0])
        proposed = {}
        mk_idx = {}
        lpq_mk = {}
        prop_ids = np.zeros(theta.shape[0])
        sigma_u = 1  # Standard deviation of auxiliary distribution

        prop_theta = theta.copy()

        # Compute model transition probabilities
        pp_mk_logZ = np.zeros(len(self.mk_logZhat.keys()))
        pp_mk_keys = []
        for i, (pmk, pmk_logZ) in enumerate(self.mk_logZhat.items()):
            pp_mk_logZ[i] = pmk_logZ
            pp_mk_keys.append(pmk)
        pp_mk_log_prob = pp_mk_logZ - logsumexp(pp_mk_logZ)

        # Process each model separately
        model_enumeration, rev = self.pmodel.enumerateModels(theta)
        for mk, mk_row_idx in model_enumeration.items():
            mk_theta = theta[mk_row_idx]
            mk_n = mk_theta.shape[0]
            params, splitdict = self.explodeParameters(theta, mk)
            betas = np.column_stack([params[xn] for xn in self.betanames])  # Active coefficients
            gammas = np.column_stack([params[xn] for xn in self.gammanames])  # All gamma indicators
            gamma_vec = gammas[0]  # Gamma vector (identical across particles)

            # Transform to base distribution
            Tmktheta, lpq1 = self.transformToBase(mk_theta, mk)

            # Compute model transition probabilities
            this_log_prob = self.mk_logZhat[mk] - logsumexp(pp_mk_logZ)
            pidx = np.random.choice(np.arange(pp_mk_logZ.shape[0]), p=np.exp(pp_mk_log_prob), size=mk_n)
            lpq_mk[mk] = this_log_prob - pp_mk_log_prob[pidx]

            proposed[mk], mk_idx[mk] = self.explodeParameters(Tmktheta, mk)

            prop_mk_theta = mk_theta.copy()
            mk_prop_ids = np.zeros(mk_n)

            # Apply proposals based on target model
            for p_i in np.unique(pidx):
                tn = (pidx == p_i).sum()
                new_mk = pp_mk_keys[p_i]

                at_idx = pidx == p_i
                if new_mk == mk:
                    # Within-model update
                    (
                        prop_mk_theta[at_idx],
                        lpq_mk[mk][at_idx],
                        mk_prop_ids[at_idx],
                    ) = self.within_model_proposal.draw(mk_theta[at_idx], tn)
                    continue

                # get column ids for toggled on and off blocks
                on_idx, off_idx = self.toggleIDX(mk, new_mk)
                # toggle ons
                for idx in on_idx:
                    Tmktheta[at_idx] = self.setVariable(Tmktheta[at_idx], self.gammanames[idx], np.ones(tn))
                # for toggle off
                for idx in off_idx:
                    Tmktheta[at_idx] = self.setVariable(Tmktheta[at_idx], self.gammanames[idx], np.zeros(tn))
                # transform back
                prop_mk_theta[at_idx], lpq2 = self.transformFromBase(Tmktheta[at_idx], new_mk)
                lpq_mk[mk][at_idx] += lpq1[at_idx] + lpq2
                mk_prop_ids[at_idx] = np.full(tn, id(self))

            prop_theta[mk_row_idx] = prop_mk_theta
            logpqratio[mk_row_idx] = lpq_mk[mk]
            prop_ids[mk_row_idx] = mk_prop_ids

        return prop_theta, logpqratio, prop_ids
