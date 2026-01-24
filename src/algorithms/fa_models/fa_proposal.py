import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import norm

from src.proposals import Proposal
from src.transforms import (
    CauchyCDF,
    CompositeTransform,
    InverseTransform,
    LogTransform,
    Sigmoid,
)


class RJFlowGlobalFactorAnalysisProposal(Proposal):
    def __init__(
        self, problem, *, indicator_name, betaii_names, betaij_names, lambda_names, within_model_proposal, **kwargs
    ):
        self.problem = problem

        self.indicator_name = indicator_name
        self.betaii_names = betaii_names
        self.betaij_names = betaij_names
        self.lambda_names = lambda_names

        assert isinstance(within_model_proposal, Proposal)
        self.within_model_proposal = within_model_proposal
        self.rv_names = betaii_names + betaij_names + lambda_names + [indicator_name]

        self.save_flows_dir = kwargs.get("save_flows_dir", "")

        super().__init__([*self.rv_names, self.within_model_proposal])

        self.exclude_concat = [indicator_name]

    def calibratemmmpd(self, mmmpd, size, t):
        self.within_model_proposal.calibratemmmpd(mmmpd, size, t)

        mklist = self.pmodel.getModelKeys()  # get all keys

        orig_theta, orig_theta_w = mmmpd.getOriginalParticleDensityForTemperature(t, resample=False)
        orig_theta_w = np.exp(orig_theta_w)
        orig_mkdict, _ = self.pmodel.enumerateModels(orig_theta)

        self.flows = {}
        self.mk_logZhat = {}

        for mk in mklist:
            self.mk_logZhat[mk] = -np.log(len(mklist))

            # use original particles at temperature t.
            mk_theta = orig_theta[orig_mkdict[mk]]
            mk_theta_w = orig_theta_w[orig_mkdict[mk]]
            mk_theta_w = np.exp(np.log(mk_theta_w) - logsumexp(np.log(mk_theta_w)))

            X, concat_indices = self.concatParameters(mk_theta, mk, return_indices=True)

            # transform the betaii variables and lambda variables using log first
            spec = {}
            dim = 0
            for rvn, rv_indices in self.pmodel.generateRVIndices(model_key=mk, flatten_tree=True).items():
                if len(rv_indices) > 0:
                    dim += len(rv_indices)
                    if rvn in self.betaii_names or rvn in self.lambda_names:
                        for i in rv_indices:
                            j = concat_indices.index(i)
                            spec[j] = CompositeTransform(
                                [
                                    LogTransform(),
                                    CauchyCDF(),
                                    InverseTransform(Sigmoid(temperature=np.sqrt(8 / np.pi))),
                                ]
                            )
                    elif rvn in self.betaij_names:
                        for i in rv_indices:
                            j = concat_indices.index(i)
                            spec[j] = CompositeTransform(
                                [
                                    CauchyCDF(),
                                    InverseTransform(Sigmoid(temperature=np.sqrt(8 / np.pi))),
                                ]
                            )

            dim -= 1
            if ~np.any(np.isfinite(np.std(X, axis=0))):
                raise ValueError("X is singular", X)

            self._calibratemmmpd(mk, mk_theta_w, spec, dim, X)

    def _calibratemmmpd(self, mk, mk_theta_w, spec, dim, X):
        raise NotImplementedError

    def transformToBase(self, inputs, mk):
        if self.getModelDim(mk) == 0:
            return inputs, np.zeros(inputs.shape[0])
        X = self.concatParameters(inputs, mk)
        XX, logdet = self.flows[mk]._transform.forward(torch.tensor(X, dtype=torch.float32))
        return self.deconcatParameters(XX.detach().numpy(), inputs, mk), logdet.detach().numpy()

    def transformFromBase(self, inputs, mk):
        if self.getModelDim(mk) == 0:
            return inputs, np.zeros(inputs.shape[0])
        X = self.concatParameters(inputs, mk)
        XX, logdet = self.flows[mk]._transform.inverse(torch.tensor(X, dtype=torch.float32))
        return self.deconcatParameters(XX.detach().numpy(), inputs, mk), logdet.detach().numpy()

    def getModelDim(self, mk):
        dim = 0
        for _, rv_indices in self.pmodel.generateRVIndices(model_key=mk, flatten_tree=True).items():
            if len(rv_indices) > 0:
                dim += len(rv_indices)
        return dim - 1

    def toggleGamma(self, mk, tidx):
        mkl = list(mk)
        mkl[tidx] = 1 - mkl[tidx]
        return tuple(mkl)

    def auxToCols(self, mk, new_mk):
        # return list of column names which are enabled from mk to new_mk
        mk_cols = self.pmodel.generateRVIndices(model_key=mk, flatten_tree=True)
        new_mk_cols = self.pmodel.generateRVIndices(model_key=new_mk, flatten_tree=True)
        return list(set(mk_cols.keys()) - set(new_mk_cols.keys()))

    def draw(self, theta, size=1):
        logpqratio = np.zeros(theta.shape[0])
        proposed = {}
        mk_idx = {}
        lpq_mk = {}
        prop_ids = np.zeros(theta.shape[0])
        sigma_u = 1

        prop_theta = theta.copy()

        pp_mk_logZ = np.zeros(len(self.mk_logZhat.keys()))
        pp_mk_keys = []
        for i, (pmk, pmk_logZ) in enumerate(self.mk_logZhat.items()):
            pp_mk_logZ[i] = pmk_logZ
            pp_mk_keys.append(pmk)
        pp_mk_log_prob = pp_mk_logZ - logsumexp(pp_mk_logZ)
        nmodels = pp_mk_logZ.shape[0]
        pp_mk_log_prob = np.zeros(nmodels) - np.log(nmodels - 1)  # hack for bartolucci estimator

        model_enumeration, _ = self.pmodel.enumerateModels(theta)

        for mk, mk_row_idx in model_enumeration.items():
            mk_theta = theta[mk_row_idx]
            mk_n = mk_theta.shape[0]

            Tmktheta, lpq1 = self.transformToBase(mk_theta, mk)
            proposed[mk], mk_idx[mk] = self.explodeParameters(Tmktheta, mk)

            this_mk_probs = np.exp(pp_mk_log_prob)
            this_p_i = pp_mk_keys.index(mk)
            this_mk_probs[this_p_i] = 0
            pidx = np.random.choice(
                np.arange(pp_mk_logZ.shape[0]), p=this_mk_probs, size=mk_n
            )  # don't do within model move. Hack for bartolucci
            lpq_mk[mk] = np.zeros(mk_n)

            prop_mk_theta = mk_theta.copy()
            mk_prop_ids = np.zeros(mk_n)

            # for each mk in pidx
            # separate theta further into model transitions
            for p_i in np.unique(pidx):
                tn = (pidx == p_i).sum()
                new_mk = pp_mk_keys[p_i]
                at_idx = pidx == p_i
                if new_mk == mk:
                    prop_mk_theta[at_idx], lpq_mk[mk][at_idx], mk_prop_ids[at_idx] = self.within_model_proposal.draw(
                        mk_theta[at_idx], tn
                    )
                    continue

                # get column ids for toggled on and off blocks
                off_cols = self.auxToCols(mk, new_mk)
                on_cols = self.auxToCols(new_mk, mk)

                # for toggle ons
                for c in on_cols:
                    u = norm(0, sigma_u).rvs(tn)
                    log_u = norm(0, sigma_u).logpdf(u)  # for naive, mean is u
                    lpq_mk[mk][at_idx] -= log_u
                    Tmktheta[at_idx] = self.setVariable(Tmktheta[at_idx], c, u)

                # for toggle offs
                for c in off_cols:
                    u = self.getVariable(Tmktheta[at_idx], c).flatten()
                    log_u = norm(0, sigma_u).logpdf(u)  # for naive, mean is u
                    lpq_mk[mk][at_idx] += log_u

                Tmktheta[at_idx] = self.setVariable(
                    Tmktheta[at_idx], self.indicator_name, np.full(tn, new_mk[0])
                )  # hack for indicator value
                # transform back
                prop_mk_theta[at_idx], lpq2 = self.transformFromBase(Tmktheta[at_idx], new_mk)

                # zero toggle offs
                for c in off_cols:
                    prop_mk_theta[at_idx] = self.setVariable(prop_mk_theta[at_idx], c, 0)

                lpq_mk[mk][at_idx] += lpq1[at_idx] + lpq2
                mk_prop_ids[at_idx] = np.full(tn, id(self))

            prop_theta[mk_row_idx] = prop_mk_theta
            logpqratio[mk_row_idx] = lpq_mk[mk]
            prop_ids[mk_row_idx] = mk_prop_ids
        return prop_theta, logpqratio, prop_ids
