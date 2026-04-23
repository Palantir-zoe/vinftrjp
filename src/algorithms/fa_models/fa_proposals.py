import numpy as np
import torch
from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from normflows.distributions.base import DiagGaussian
from scipy.special import logsumexp
from scipy.stats import gaussian_kde, multivariate_normal

from src.distributions import InvGammaDistribution
from src.flows import RationalQuadraticFlowFAV
from src.proposals import Proposal
from src.transforms import (
    ColumnSpecificTransform,
    CompositeTransform,
    FixedNorm,
    InverseTransform,
    LogTransform,
    NaiveGaussianTransform,
    Sigmoid,
)

from ..utils import train_with_checkpoint
from .fa_proposal import RJFlowGlobalFactorAnalysisProposal


class RJFlowGlobalFactorAnalysisProposalAF(RJFlowGlobalFactorAnalysisProposal):
    def __init__(self, problem, **kwargs):
        super().__init__(problem, **kwargs)

    def _calibratemmmpd(self, mk, mk_theta_w, spec, dim, X):
        bnorm = StandardNormal((dim,))

        spectransform = ColumnSpecificTransform(spec)
        X_, _ = spectransform.forward(torch.Tensor(X))

        tf = CompositeTransform(
            [
                ColumnSpecificTransform(spec),
                InverseTransform(NaiveGaussianTransform(X_, torch.Tensor(mk_theta_w))),
            ]
        )

        folder = self.__class__.__name__
        self.flows[mk] = train_with_checkpoint(self.save_flows_dir, folder, mk, Flow, tf, bnorm)


class RJFlowGlobalFactorAnalysisProposalNF(RJFlowGlobalFactorAnalysisProposal):
    def __init__(self, problem, **kwargs):
        super().__init__(problem, **kwargs)

    def _calibratemmmpd(self, mk, mk_theta_w, spec, dim, X):
        bnorm = StandardNormal((dim,))
        ls = Sigmoid(temperature=np.sqrt(8 / np.pi))

        spectransform = ColumnSpecificTransform(spec)
        X_, _ = spectransform.forward(torch.Tensor(X))

        fn = CompositeTransform(
            [
                ColumnSpecificTransform(spec),
                FixedNorm(X_, torch.Tensor(mk_theta_w)),
            ]
        )

        folder = self.__class__.__name__
        self.flows[mk] = train_with_checkpoint(
            self.save_flows_dir, folder, mk, RationalQuadraticFlowFAV.factory, X, bnorm, ls, fn, mk_theta_w
        )


class RJFlowGlobalFactorAnalysisProposalVINF(RJFlowGlobalFactorAnalysisProposal):
    def __init__(self, normalizing_flows, problem, **kwargs):
        self.normalizing_flows = normalizing_flows
        super().__init__(problem, **kwargs)

    def calibratemmmpd(self, mmmpd, size, t):
        self.within_model_proposal.calibratemmmpd(mmmpd, size, t)

        mklist = self.pmodel.getModelKeys()  # get all keys

        self.flows = {}
        self.mk_logZhat = {}

        for mk in mklist:
            self.mk_logZhat[mk] = -np.log(len(mklist))

            p = self.problem.target(self.pmodel.y_data, k=mk[0])
            q0 = DiagGaussian(p.ndim)

            folder = self.__class__.__name__
            self.flows[mk] = train_with_checkpoint(
                self.save_flows_dir, folder, mk, self.normalizing_flows, q0=q0, target=p
            )

    def transformToBase(self, inputs, mk):
        if self.getModelDim(mk) == 0:
            return inputs, np.zeros(inputs.shape[0])
        X = self.concatParameters(inputs, mk)
        XX, logdet = self.flows[mk]._transform.inverse(torch.tensor(X, dtype=torch.float32))
        return self.deconcatParameters(XX.detach().numpy(), inputs, mk), logdet.detach().numpy()

    def transformFromBase(self, inputs, mk):
        if self.getModelDim(mk) == 0:
            return inputs, np.zeros(inputs.shape[0])
        X = self.concatParameters(inputs, mk)
        XX, logdet = self.flows[mk]._transform.forward(torch.tensor(X, dtype=torch.float32))
        return self.deconcatParameters(XX.detach().numpy(), inputs, mk), logdet.detach().numpy()


class RJFlowGlobalFactorAnalysisProposalVINFRejectionFree(RJFlowGlobalFactorAnalysisProposalVINF):
    def __init__(self, normalizing_flows, problem, posterior_model_probabilities, **kwargs):
        self.posterior_model_probabilities = posterior_model_probabilities
        super().__init__(normalizing_flows, problem, **kwargs)

    def _get_model_log_probs(self):
        mklist = self.pmodel.getModelKeys()
        if self.posterior_model_probabilities is None:
            raise ValueError("posterior_model_probabilities must be provided for rejection-free FA proposal.")

        probs = np.zeros(len(mklist), dtype=np.float64)
        for i, mk in enumerate(mklist):
            probs[i] = float(self.posterior_model_probabilities.get(mk, 0.0))

        if np.any(probs < 0):
            raise ValueError(f"posterior_model_probabilities must be non-negative, got {self.posterior_model_probabilities}")
        if not np.any(probs > 0):
            raise ValueError(f"posterior_model_probabilities assign zero mass to all models: {self.posterior_model_probabilities}")

        probs = probs / probs.sum()
        return mklist, np.log(probs)

    def _sample_model_conditional(self, mk, size):
        flow = self.flows[mk]
        samples, log_q = flow.sample(size)
        x = np.asarray(samples.detach().cpu().tolist(), dtype=np.float64)
        log_q = np.asarray(log_q.detach().cpu().tolist(), dtype=np.float64)

        theta = np.zeros((size, self.pmodel.dim()), dtype=np.float64)
        k_col = self.pmodel.generateRVIndices()[self.indicator_name][0]
        theta[:, k_col] = mk[0]
        theta = self.deconcatParameters(x, theta, mk)
        theta = self.pmodel.sanitise(theta)
        return theta, log_q

    def _eval_model_conditional_log_prob(self, theta, mk):
        if self.getModelDim(mk) == 0:
            return np.zeros(theta.shape[0], dtype=np.float64)
        x = self.concatParameters(theta, mk)
        log_q = self.flows[mk].log_prob(torch.tensor(x, dtype=torch.float32))
        return np.asarray(log_q.detach().cpu().tolist(), dtype=np.float64)

    def draw(self, theta, size=1):
        prop_theta = np.zeros_like(theta)
        logpqratio = np.zeros(theta.shape[0], dtype=np.float64)
        prop_ids = np.full(theta.shape[0], id(self))

        mklist, mk_log_probs = self._get_model_log_probs()
        mk_to_index = {mk: i for i, mk in enumerate(mklist)}
        mk_probs = np.exp(mk_log_probs)

        model_enumeration, _ = self.pmodel.enumerateModels(theta)
        for mk, mk_row_idx in model_enumeration.items():
            mk_theta = theta[mk_row_idx]
            cur_log_q_model = mk_log_probs[mk_to_index[mk]]
            cur_log_q_param = self._eval_model_conditional_log_prob(mk_theta, mk)

            pidx = np.random.choice(np.arange(len(mklist)), p=mk_probs, size=mk_theta.shape[0])

            for p_i in np.unique(pidx):
                at_idx = pidx == p_i
                new_mk = mklist[p_i]
                tn = int(at_idx.sum())

                prop_block, prop_log_q_param = self._sample_model_conditional(new_mk, tn)
                prop_theta[mk_row_idx[at_idx]] = prop_block

                logpqratio[mk_row_idx[at_idx]] = (
                    cur_log_q_model
                    + cur_log_q_param[at_idx]
                    - mk_log_probs[p_i]
                    - prop_log_q_param
                )

        return prop_theta, logpqratio, prop_ids


class FARWProposal(Proposal):
    def __init__(self, problem, *, betaii_names, betaij_names, lambda_names, **kwargs):
        self.problem = problem

        self.betaii_names = betaii_names
        self.betaij_names = betaij_names
        self.lambda_names = lambda_names

        super().__init__(betaii_names + betaij_names + lambda_names)

    def transformToBase(self, inputs, mk):
        X = self.concatParameters(inputs, mk)
        XX, logdet = self.flow.forward(torch.tensor(X, dtype=torch.float32))
        return XX.detach().numpy(), logdet.detach().numpy()

    def transformFromBase(self, X, inputs, mk):
        XX, logdet = self.flow.inverse(torch.tensor(X, dtype=torch.float32))
        return self.deconcatParameters(XX.detach().numpy(), inputs, mk), logdet.detach().numpy()

    def calibratemmmpd(self, mmmpd, size, t):
        mk = self.getModelIdentifier()
        mk_theta, mk_theta_w = mmmpd.getOriginalParticleDensityForTemperature(t, resample=True)
        mk_theta_w = np.exp(mk_theta_w - logsumexp(mk_theta_w))

        _, concat_indices = self.concatParameters(mk_theta, mk, return_indices=True)

        spec = {}
        dim = 0
        for rvn, rv_indices in self.pmodel.generateRVIndices(model_key=mk, flatten_tree=True).items():
            if len(rv_indices) > 0:
                dim += len(rv_indices)
                if rvn in self.betaii_names or rvn in self.lambda_names:
                    for i in rv_indices:
                        j = concat_indices.index(i)
                        spec[j] = LogTransform()
        self.flow = ColumnSpecificTransform(spec)

        Tk_theta, _ = self.transformToBase(mk_theta, mk)
        self.cov = np.cov(Tk_theta.T)
        self.propscale = 0.05  # starting point

    def draw(self, theta, size=1):
        mk = self.getModelIdentifier()
        N = theta.shape[0]
        TkX, lpq1 = self.transformToBase(theta, mk)
        d = TkX.shape[1]
        propTkX = TkX + multivariate_normal(np.zeros(d), self.cov * self.propscale).rvs(N)
        proptheta, lpq2 = self.transformFromBase(propTkX, theta.copy(), mk)
        return proptheta, lpq1 + lpq2, np.full(N, id(self))


class RJGlobalFactorAnalysisProposalLW(Proposal):
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

        super().__init__([*self.rv_names, self.within_model_proposal])

        self.exclude_concat = [indicator_name]

    def calibratemmmpd(self, mmmpd, size, t):
        self.within_model_proposal.calibratemmmpd(mmmpd, size, t)

        mklist = self.pmodel.getModelKeys()  # get all keys

        orig_theta, _ = mmmpd.getOriginalParticleDensityForTemperature(t, resample=True, resample_max_size=10000)
        orig_mkdict, _ = self.pmodel.enumerateModels(orig_theta)

        self.flows = {}
        self.mk_logZhat = {}
        self.beta_col_names = {}
        self.beta_cov = {}
        self.beta_mean = {}
        self.lambda_a = {}
        self.lambda_scale = {}
        for mk in mklist:
            mk_theta, mk_theta_w = mmmpd.getParticleDensityForModelAndTemperature(mk, t, resample=False)
            mk_theta_w = np.exp(np.log(mk_theta_w) - logsumexp(np.log(mk_theta_w)))
            if mk in orig_mkdict.keys():
                self.mk_logZhat[mk] = np.log(orig_mkdict[mk].shape[0] * 1.0 / 10000)
            else:
                self.mk_logZhat[mk] = np.inf

            self.beta_col_names[mk] = []
            for rvn, rv_indices in self.pmodel.generateRVIndices(model_key=mk, flatten_tree=True).items():
                if len(rv_indices) > 0:
                    if rvn in self.betaii_names or rvn in self.betaij_names:
                        self.beta_col_names[mk].append(rvn)

            beta_all = np.zeros((mk_theta.shape[0], len(self.beta_col_names[mk])))
            for i, c in enumerate(self.beta_col_names[mk]):
                if c in self.betaii_names:
                    beta_all[:, i] = np.log(self.getVariable(mk_theta, c).flatten())
                else:
                    beta_all[:, i] = self.getVariable(mk_theta, c).flatten()
            self.beta_cov[mk] = 2 * np.cov(beta_all.T, aweights=mk_theta_w)
            self.beta_mean[mk] = np.average(beta_all, axis=0, weights=mk_theta_w)

            self.lambda_a[mk] = {}
            self.lambda_scale[mk] = {}
            for _, c in enumerate(self.lambda_names):
                this_lambda = self.getVariable(mk_theta, c).flatten()

                h = gaussian_kde(this_lambda).pdf(this_lambda)
                mode = this_lambda[np.argmax(h)]

                self.lambda_a[mk][c] = 18
                self.lambda_scale[mk][c] = 18 * mode

    def draw(self, theta, size=1):
        logpqratio = np.zeros(theta.shape[0])
        lpq_mk = {}
        prop_ids = np.zeros(theta.shape[0])

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

                # the Lopes & West RJ proposal is an independence proposal
                # evaluate old parameters
                old_betas = np.zeros((tn, len(self.beta_col_names[mk])))
                for vi, c in enumerate(self.beta_col_names[mk]):
                    if c in self.betaii_names:
                        old_betas[:, vi] = np.log(self.getVariable(mk_theta[at_idx], c).flatten())
                        lpq_mk[mk][at_idx] += -old_betas[:, vi]
                    else:
                        old_betas[:, vi] = self.getVariable(mk_theta[at_idx], c).flatten()
                lpq_mk[mk][at_idx] += multivariate_normal(self.beta_mean[mk], self.beta_cov[mk]).logpdf(old_betas)
                old_lambdas = np.zeros((tn, len(self.lambda_names)))
                for vi, c in enumerate(self.lambda_names):
                    old_lambdas[:, vi] = self.getVariable(mk_theta[at_idx], c).flatten()
                    lpq_mk[mk][at_idx] += InvGammaDistribution(self.lambda_a[mk][c], self.lambda_scale[mk][c]).logeval(
                        old_lambdas[:, vi]
                    )

                # We draw betas from a fitted MVN, and lambdas from IG(1.1,0.05) priors
                new_betas = (
                    multivariate_normal(self.beta_mean[new_mk], self.beta_cov[new_mk])
                    .rvs(tn)
                    .reshape((tn, self.beta_mean[new_mk].shape[0]))
                )
                lpq_mk[mk][at_idx] -= multivariate_normal(self.beta_mean[new_mk], self.beta_cov[new_mk]).logpdf(
                    new_betas
                )
                for vi, c in enumerate(self.beta_col_names[new_mk]):
                    if c in self.betaii_names:
                        prop_mk_theta[at_idx] = self.setVariable(prop_mk_theta[at_idx], c, np.exp(new_betas[:, vi]))
                        lpq_mk[mk][at_idx] -= -new_betas[:, vi]
                    else:
                        prop_mk_theta[at_idx] = self.setVariable(prop_mk_theta[at_idx], c, new_betas[:, vi])

                for _, c in enumerate(self.lambda_names):
                    new_lambdas = InvGammaDistribution(self.lambda_a[new_mk][c], self.lambda_scale[new_mk][c]).draw(
                        size=tn
                    )
                    lpq_mk[mk][at_idx] -= InvGammaDistribution(
                        self.lambda_a[new_mk][c], self.lambda_scale[new_mk][c]
                    ).logeval(new_lambdas)
                    prop_mk_theta[at_idx] = self.setVariable(prop_mk_theta[at_idx], c, new_lambdas)

                prop_mk_theta[at_idx] = self.setVariable(
                    prop_mk_theta[at_idx], self.indicator_name, np.full(tn, new_mk[0])
                )

                mk_prop_ids[at_idx] = np.full(tn, id(self))

            prop_theta[mk_row_idx] = prop_mk_theta
            logpqratio[mk_row_idx] = lpq_mk[mk]
            prop_ids[mk_row_idx] = mk_prop_ids
        return prop_theta, logpqratio, prop_ids
