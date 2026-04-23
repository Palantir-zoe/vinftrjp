import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm

from src.proposals import Proposal

from ..utils import train_with_checkpoint


class ChangePointWithinModelProposal(Proposal):
    def __init__(self, *, problem, segment_names, proposal_scale=0.35, **kwargs):
        self.problem = problem
        self.segment_names = segment_names
        self.proposal_scale = proposal_scale
        self.cov = None

        super().__init__(segment_names)

    def _regularize_covariance(self, cov, dim):
        if dim == 0:
            return np.zeros((0, 0), dtype=np.float64)

        cov = np.asarray(cov, dtype=np.float64)
        if dim == 1:
            var = float(np.squeeze(cov))
            if not np.isfinite(var) or var <= 0:
                var = 1.0
            return np.array([[var]], dtype=np.float64)

        cov = np.atleast_2d(cov)
        if cov.shape != (dim, dim) or not np.isfinite(cov).all():
            cov = np.eye(dim, dtype=np.float64)

        cov = 0.5 * (cov + cov.T)
        min_eig = float(np.min(np.linalg.eigvalsh(cov)))
        if min_eig < 1e-8:
            cov = cov + np.eye(dim, dtype=np.float64) * (1e-8 - min_eig)
        return cov

    def calibratemmmpd(self, mmmpd, size, t):
        mk = self.getModelIdentifier()
        theta, theta_w = mmmpd.getOriginalParticleDensityForTemperature(t, resample=False)
        dim = sum(len(param_range) for _, param_range in self.rv_indices[mk].items() if len(param_range) > 0)

        if dim == 0:
            self.cov = np.zeros((0, 0), dtype=np.float64)
            return

        model_key_indices, _ = self.getModel().enumerateModels(theta)
        if mk not in model_key_indices:
            self.cov = np.eye(dim, dtype=np.float64)
            return

        mk_idx = model_key_indices[mk]
        X = self.concatParameters(theta[mk_idx], mk)
        mk_theta_w = theta_w[mk_idx]
        mk_theta_w = np.exp(mk_theta_w - logsumexp(mk_theta_w))

        if X.shape[0] <= 1:
            cov = np.eye(dim, dtype=np.float64)
        else:
            cov = np.cov(X.T, aweights=mk_theta_w)

        self.cov = self._regularize_covariance(cov, dim)

    def draw(self, theta, size=1):
        mk = self.getModelIdentifier()
        n = theta.shape[0]
        x = self.concatParameters(theta, mk)
        dim = x.shape[1]

        if dim == 0:
            return theta.copy(), np.zeros(n), np.full(n, id(self))

        base_cov = self.cov if self.cov is not None else np.eye(dim, dtype=np.float64)
        prop_cov = self._regularize_covariance(base_cov * self.proposal_scale, dim)

        if dim == 1:
            std = np.sqrt(float(prop_cov[0, 0]))
            noise = norm(0, std).rvs(n).reshape(n, 1)
        else:
            noise = multivariate_normal(np.zeros(dim), prop_cov).rvs(n)
            noise = np.asarray(noise, dtype=np.float64).reshape(n, dim)

        prop_x = x + noise
        prop_theta = self.deconcatParameters(prop_x, theta.copy(), mk)
        return prop_theta, np.zeros(n), np.full(n, id(self))


class RJFlowGlobalChangePointProposalVINF(Proposal):
    def __init__(
        self,
        *,
        normalizing_flows,
        problem,
        indicator_name,
        segment_names,
        within_model_proposal,
        within_model_prob=0.5,
        within_model_scale=0.35,
        aux_scale=1.0,
        save_flows_dir="",
        use_conditional_shared_flow=True,
        **kwargs,
    ):
        self.normalizing_flows = normalizing_flows
        self.problem = problem

        self.indicator_name = indicator_name
        self.segment_names = segment_names

        assert isinstance(within_model_proposal, Proposal)
        self.within_model_proposal = within_model_proposal
        self.within_model_prob = within_model_prob
        self.within_model_scale = within_model_scale
        self.aux_scale = aux_scale
        self.save_flows_dir = save_flows_dir
        self.use_conditional_shared_flow = use_conditional_shared_flow

        self.rv_names = self.segment_names + [self.indicator_name]

        super().__init__([*self.rv_names, self.within_model_proposal])

        self.exclude_concat = [self.indicator_name]

    def calibratemmmpd(self, mmmpd, size, t):
        self.within_model_proposal.calibratemmmpd(mmmpd, size, t)

        self.flows = {}
        self.mk_logZhat = {}
        mklist = self.pmodel.getModelKeys()

        if self.use_conditional_shared_flow:
            target = self.problem.conditional_target(reference_scale=self.aux_scale)
            folder = f"{self.__class__.__name__}_collapsed_ctp_v1"
            self.shared_flow = train_with_checkpoint(
                self.save_flows_dir,
                folder,
                ("shared", self.problem.k_max),
                self.normalizing_flows,
                target=target,
            ).double()
            self.conditional_target = target
            for mk in mklist:
                self.mk_logZhat[mk] = -np.log(len(mklist))
                self.flows[mk] = self.shared_flow
            return

        for mk in mklist:
            self.mk_logZhat[mk] = -np.log(len(mklist))
            target = self.problem.target(k=mk[0])

            if target.ndim == 0:
                self.flows[mk] = None
                continue

            folder = f"{self.__class__.__name__}_collapsed_trj_v1"
            self.flows[mk] = train_with_checkpoint(
                self.save_flows_dir,
                folder,
                mk,
                self.normalizing_flows,
                target=target,
            ).double()

    def transformToBase(self, inputs, mk):
        if self.use_conditional_shared_flow:
            xx_np, logdet_np, _ = self._transform_to_base_matrix(inputs, mk)
            return xx_np, logdet_np

        xx_np, logdet_np = self._transform_to_base_matrix(inputs, mk)
        return self.deconcatParameters(xx_np, inputs, mk), logdet_np

    def transformFromBase(self, inputs, mk):
        x = self.concatParameters(inputs, mk)
        if self.use_conditional_shared_flow:
            return self._transform_from_base_matrix(x, inputs, mk)[:2]

        xx_np, logdet_np = self._transform_from_base_matrix(x, inputs, mk)
        return self.deconcatParameters(xx_np, inputs, mk), logdet_np

    def _transform_to_base_matrix(self, inputs, mk):
        if self.use_conditional_shared_flow:
            x_aug, inactive = self._pack_augmented_parameters(inputs, mk)
            context = self._context_for_model(mk, inputs.shape[0], dtype=torch.float64)
            xx, logdet = self.shared_flow._transform.inverse(
                torch.tensor(x_aug, dtype=torch.float64),
                context=context,
            )
            xx_np = np.asarray(xx.detach().cpu().tolist(), dtype=np.float64)
            logdet_np = np.asarray(logdet.detach().cpu().tolist(), dtype=np.float64)
            inactive_log_prob = self._reference_log_prob_np(inactive)
            return xx_np, logdet_np, inactive_log_prob

        x = self.concatParameters(inputs, mk)
        if x.shape[1] == 0:
            return x.copy(), np.zeros(inputs.shape[0], dtype=np.float64)
        xx, logdet = self.flows[mk]._transform.inverse(torch.tensor(x, dtype=torch.float64))
        xx_np = np.asarray(xx.detach().cpu().tolist(), dtype=np.float64)
        logdet_np = np.asarray(logdet.detach().cpu().tolist(), dtype=np.float64)
        return xx_np, logdet_np

    def _transform_from_base_matrix(self, x, inputs, mk):
        if self.use_conditional_shared_flow:
            context = self._context_for_model(mk, x.shape[0], dtype=torch.float64)
            xx, logdet = self.shared_flow._transform.forward(
                torch.tensor(x, dtype=torch.float64),
                context=context,
            )
            xx_np = np.asarray(xx.detach().cpu().tolist(), dtype=np.float64)
            logdet_np = np.asarray(logdet.detach().cpu().tolist(), dtype=np.float64)
            prop_theta, inactive = self._unpack_augmented_parameters(xx_np, inputs, mk)
            inactive_log_prob = self._reference_log_prob_np(inactive)
            return prop_theta, logdet_np, inactive_log_prob

        if x.shape[1] == 0:
            return x.copy(), np.zeros(inputs.shape[0], dtype=np.float64)
        xx, logdet = self.flows[mk]._transform.forward(torch.tensor(x, dtype=torch.float64))
        xx_np = np.asarray(xx.detach().cpu().tolist(), dtype=np.float64)
        logdet_np = np.asarray(logdet.detach().cpu().tolist(), dtype=np.float64)
        return xx_np, logdet_np

    def _parameter_names_for_model(self, mk):
        active_names = []
        for name in self.segment_names:
            if name in self.rv_indices[mk]:
                active_names.append(name)
        return active_names

    def _jump_distribution(self, k):
        if k <= 0:
            return {1: 1.0}
        if k >= self.problem.k_max:
            return {self.problem.k_max - 1: 1.0}
        return {k - 1: 0.5, k + 1: 0.5}

    def _model_log_ratio(self, cur_k, new_k):
        q_forward = self._jump_distribution(cur_k)[new_k]
        q_reverse = self._jump_distribution(new_k)[cur_k]
        return np.log(q_reverse) - np.log(q_forward)

    def _context_for_model(self, mk, n, dtype=torch.float64):
        context = torch.zeros((n, self.problem.k_max), dtype=dtype)
        k = int(mk[0])
        if k > 0:
            context[:, :k] = 1.0
        return context

    def _reference_log_prob_np(self, x):
        if x.shape[1] == 0:
            return np.zeros(x.shape[0], dtype=np.float64)
        scale = float(self.aux_scale)
        return (-np.log(scale) - 0.5 * np.log(2.0 * np.pi) - 0.5 * (x / scale) ** 2).sum(axis=1)

    def _pack_augmented_parameters(self, theta, mk):
        k = int(mk[0])
        active = self.concatParameters(theta, mk)
        n = theta.shape[0]
        inactive_dim = self.problem.k_max - k
        if inactive_dim > 0:
            inactive = norm(0, self.aux_scale).rvs((n, inactive_dim))
            inactive = np.asarray(inactive, dtype=np.float64).reshape(n, inactive_dim)
            return np.column_stack([active, inactive]), inactive
        return active.copy(), np.zeros((n, 0), dtype=np.float64)

    def _unpack_augmented_parameters(self, x_aug, theta, mk):
        k = int(mk[0])
        active = x_aug[:, :k]
        inactive = x_aug[:, k:]
        prop_theta = self.deconcatParameters(active, theta, mk)
        return prop_theta, inactive

    def sample_model_parameters_from_flow(self, mk, size):
        theta = np.zeros((size, self.pmodel.dim()), dtype=np.float64)
        theta = self.setVariable(theta, self.indicator_name, np.full(size, int(mk[0])))

        if self.use_conditional_shared_flow:
            context = self._context_for_model(mk, size, dtype=torch.float64)
            samples, _ = self.shared_flow.sample(size, context=context)
            x_aug = np.asarray(samples.detach().cpu().tolist(), dtype=np.float64)
            theta, _ = self._unpack_augmented_parameters(x_aug, theta, mk)
            return self.pmodel.sanitise(theta)

        flow = self.flows[mk]
        if flow is not None:
            samples, _ = flow.sample(size)
            x = np.asarray(samples.detach().cpu().tolist(), dtype=np.float64)
            theta = self.deconcatParameters(x, theta, mk)
        return self.pmodel.sanitise(theta)

    def _append_birth_coordinate(self, x, aux):
        return np.column_stack([x, aux])

    def _drop_death_coordinate(self, x):
        return x[:, :-1].copy(), x[:, -1].copy()

    def draw(self, theta, size=1):
        prop_theta = theta.copy()
        logpqratio = np.zeros(theta.shape[0])
        prop_ids = np.full(theta.shape[0], id(self))

        model_enumeration, _ = self.pmodel.enumerateModels(theta)
        for mk, mk_row_idx in model_enumeration.items():
            mk_theta = theta[mk_row_idx]
            k = int(mk[0])

            mk_logpq = np.zeros(mk_theta.shape[0])
            prop_mk_theta = mk_theta.copy()
            mk_prop_ids = np.full(mk_theta.shape[0], id(self))

            within_idx = np.random.rand(mk_theta.shape[0]) < self.within_model_prob
            jump_idx = ~within_idx

            if within_idx.any():
                (
                    prop_mk_theta[within_idx],
                    mk_logpq[within_idx],
                    mk_prop_ids[within_idx],
                ) = self.within_model_proposal.draw(mk_theta[within_idx], within_idx.sum())

            if jump_idx.any():
                if self.use_conditional_shared_flow:
                    jump_x, logdet_to_base, forward_aux_log_prob = self._transform_to_base_matrix(
                        mk_theta[jump_idx], mk
                    )
                else:
                    jump_x, logdet_to_base = self._transform_to_base_matrix(mk_theta[jump_idx], mk)
                    forward_aux_log_prob = np.zeros(jump_idx.sum(), dtype=np.float64)
                jump_distribution = self._jump_distribution(k)
                choices = np.array(list(jump_distribution.keys()), dtype=int)
                probs = np.array(list(jump_distribution.values()), dtype=float)
                proposed_k = np.random.choice(choices, p=probs, size=jump_idx.sum())

                jump_logpq = np.zeros(jump_idx.sum())
                jump_prop_theta = mk_theta[jump_idx].copy()

                for new_k in np.unique(proposed_k):
                    local_idx = proposed_k == new_k
                    new_mk = (int(new_k),)
                    tn = local_idx.sum()

                    block_x = jump_x[local_idx].copy()
                    block_theta = mk_theta[jump_idx][local_idx].copy()

                    if self.use_conditional_shared_flow:
                        block_theta = self.setVariable(block_theta, self.indicator_name, np.full(tn, new_k))
                        prop_jump, logdet_from_base, reverse_aux_log_prob = self._transform_from_base_matrix(
                            block_x, block_theta, new_mk
                        )
                        jump_logpq[local_idx] += reverse_aux_log_prob - forward_aux_log_prob[local_idx]
                    elif new_k == k + 1:
                        aux = norm(0, self.aux_scale).rvs(tn)
                        prop_x = self._append_birth_coordinate(block_x, aux)
                        block_theta = self.setVariable(block_theta, self.indicator_name, np.full(tn, new_k))
                        prop_concat, logdet_from_base = self._transform_from_base_matrix(prop_x, block_theta, new_mk)
                        prop_jump = self.deconcatParameters(prop_concat, block_theta, new_mk)
                        jump_logpq[local_idx] -= norm(0, self.aux_scale).logpdf(aux)
                    elif new_k == k - 1:
                        prop_x, removed = self._drop_death_coordinate(block_x)
                        block_theta = self.setVariable(block_theta, self.indicator_name, np.full(tn, new_k))
                        prop_concat, logdet_from_base = self._transform_from_base_matrix(prop_x, block_theta, new_mk)
                        prop_jump = self.deconcatParameters(prop_concat, block_theta, new_mk)
                        jump_logpq[local_idx] += norm(0, self.aux_scale).logpdf(removed)
                    else:
                        raise ValueError(f"Unsupported jump from {k} to {new_k}")

                    jump_prop_theta[local_idx] = prop_jump
                    jump_logpq[local_idx] += self._model_log_ratio(k, int(new_k)) + logdet_from_base

                prop_mk_theta[jump_idx] = jump_prop_theta
                mk_logpq[jump_idx] = jump_logpq + logdet_to_base

            prop_theta[mk_row_idx] = prop_mk_theta
            logpqratio[mk_row_idx] = mk_logpq
            prop_ids[mk_row_idx] = mk_prop_ids

        return prop_theta, logpqratio, prop_ids
