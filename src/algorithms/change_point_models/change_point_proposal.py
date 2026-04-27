import numpy as np
import torch
from nflows.distributions.normal import StandardNormal
from scipy.special import logsumexp
from scipy.stats import beta as beta_distribution
from scipy.stats import multivariate_normal, norm
import time

from src.flows import ConditionalMaskedRationalQuadraticFlow
from src.proposals import Proposal
from src.transforms import CauchyCDF, FixedNorm

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
        between_model_move="ctp",
        split_beta=2.0,
        semantic_fit_samples=2048,
        semantic_min_sigma=0.25,
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
        self.between_model_move = between_model_move
        self.split_beta = float(split_beta)
        self.semantic_fit_samples = int(semantic_fit_samples)
        self.semantic_min_sigma = float(semantic_min_sigma)
        if self.between_model_move not in {"ctp", "latent", "semantic", "semantic_learned"}:
            raise ValueError(
                "between_model_move must be one of {'ctp', 'latent', 'semantic', 'semantic_learned'}, "
                f"got {self.between_model_move}"
            )

        self.rv_names = self.segment_names + [self.indicator_name]

        super().__init__([*self.rv_names, self.within_model_proposal])

        self.exclude_concat = [self.indicator_name]

    def calibratemmmpd(self, mmmpd, size, t):
        within_start = time.perf_counter()
        self.within_model_proposal.calibratemmmpd(mmmpd, size, t)
        self.within_model_calibration_seconds = time.perf_counter() - within_start
        td_start = time.perf_counter()

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
            if self.between_model_move == "semantic_learned":
                self._fit_semantic_rho_models()
            self.td_calibration_seconds = time.perf_counter() - td_start
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

        if self.between_model_move == "semantic_learned":
            self._fit_semantic_rho_models()
        self.td_calibration_seconds = time.perf_counter() - td_start

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

    def _logsigmoid_np(self, x):
        return -np.logaddexp(0.0, -x)

    def _raw_to_log_weights(self, x):
        x = np.asarray(x, dtype=np.float64)
        n, k = x.shape
        if k == 0:
            return np.zeros((n, 1), dtype=np.float64)

        offsets = (k + 1) - np.arange(1, k + 1, dtype=np.float64)
        logits = x - np.log(offsets)
        log_z = self._logsigmoid_np(logits)
        log_one_minus_z = self._logsigmoid_np(-logits)
        prefix = np.cumsum(log_one_minus_z, axis=1)
        prev_prefix = np.column_stack([np.zeros(n, dtype=np.float64), prefix[:, :-1]])
        return np.column_stack([prev_prefix + log_z, prefix[:, -1]])

    def _raw_to_weights(self, x):
        log_weights = self._raw_to_log_weights(x)
        log_weights = log_weights - log_weights.max(axis=1, keepdims=True)
        weights = np.exp(log_weights)
        weights = weights / weights.sum(axis=1, keepdims=True)
        return np.clip(weights, 1e-300, 1.0)

    def _weights_to_raw(self, weights):
        weights = np.asarray(weights, dtype=np.float64)
        n, dim = weights.shape
        k = dim - 1
        if k == 0:
            return np.zeros((n, 0), dtype=np.float64)

        weights = np.clip(weights, 1e-300, 1.0)
        weights = weights / weights.sum(axis=1, keepdims=True)
        raw = np.zeros((n, k), dtype=np.float64)
        remaining = np.ones(n, dtype=np.float64)
        offsets = (k + 1) - np.arange(1, k + 1, dtype=np.float64)

        for i in range(k):
            z = np.clip(weights[:, i] / remaining, 1e-12, 1.0 - 1e-12)
            raw[:, i] = np.log(z) - np.log1p(-z) + np.log(offsets[i])
            remaining = np.clip(remaining - weights[:, i], 1e-300, 1.0)

        return raw

    def _raw_to_weight_logdet(self, x):
        x = np.asarray(x, dtype=np.float64)
        n, k = x.shape
        if k == 0:
            return np.zeros(n, dtype=np.float64)

        offsets = (k + 1) - np.arange(1, k + 1, dtype=np.float64)
        logits = x - np.log(offsets)
        log_weights = self._raw_to_log_weights(x)
        return (-logits + self._logsigmoid_np(logits) + log_weights[:, :-1]).sum(axis=1)

    def _split_selection_log_prob(self, weights, split_idx):
        return np.log(np.clip(weights[np.arange(weights.shape[0]), split_idx], 1e-300, 1.0))

    def _merge_selection_log_prob(self, k, n):
        return np.full(n, -np.log(k), dtype=np.float64)

    def _logit_np(self, p):
        p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
        return np.log(p) - np.log1p(-p)

    def _sigmoid_np(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _fit_semantic_rho_models(self):
        self.semantic_rho_models = {}

        for parent_k in range(self.problem.k_max):
            child_mk = (parent_k + 1,)
            child_theta = self.sample_model_parameters_from_flow(child_mk, self.semantic_fit_samples)
            child_x = self.concatParameters(child_theta, child_mk)
            child_weights = self._raw_to_weights(child_x)

            for split_idx in range(parent_k + 1):
                parent_weights = np.zeros((child_weights.shape[0], parent_k + 1), dtype=np.float64)
                merged = child_weights[:, split_idx] + child_weights[:, split_idx + 1]
                rho = child_weights[:, split_idx] / np.clip(merged, 1e-300, 1.0)
                logit_rho = self._logit_np(rho)

                parent_weights[:, :split_idx] = child_weights[:, :split_idx]
                parent_weights[:, split_idx] = merged
                parent_weights[:, split_idx + 1 :] = child_weights[:, split_idx + 2 :]

                parent_x = self._weights_to_raw(parent_weights)
                X = np.column_stack([np.ones(parent_x.shape[0], dtype=np.float64), parent_x])
                ridge = 1e-6 * np.eye(X.shape[1], dtype=np.float64)
                coef = np.linalg.solve(X.T @ X + ridge, X.T @ logit_rho)
                resid = logit_rho - X @ coef
                sigma = max(self.semantic_min_sigma, float(np.std(resid)))

                self.semantic_rho_models[(parent_k, split_idx)] = {
                    "coef": coef,
                    "sigma": sigma,
                }

    def _semantic_learned_logit_params(self, parent_x, parent_k, split_idx):
        model = getattr(self, "semantic_rho_models", {}).get((parent_k, split_idx))
        if model is None:
            sigma = max(self.semantic_min_sigma, 1.0)
            return np.zeros(parent_x.shape[0], dtype=np.float64), np.full(parent_x.shape[0], sigma, dtype=np.float64)

        X = np.column_stack([np.ones(parent_x.shape[0], dtype=np.float64), parent_x])
        mu = X @ model["coef"]
        sigma = np.full(parent_x.shape[0], model["sigma"], dtype=np.float64)
        return mu, sigma

    def _semantic_learned_logpdf(self, rho, parent_x, parent_k, split_idx):
        mu, sigma = self._semantic_learned_logit_params(parent_x, parent_k, split_idx)
        logit_rho = self._logit_np(rho)
        return norm(mu, sigma).logpdf(logit_rho) - np.log(np.clip(rho * (1.0 - rho), 1e-300, 1.0))

    def _semantic_birth(self, block_theta, mk, new_mk):
        k = int(mk[0])
        n = block_theta.shape[0]
        x = self.concatParameters(block_theta, mk)
        weights = self._raw_to_weights(x)

        split_idx = np.zeros(n, dtype=int)
        for row in range(n):
            split_idx[row] = np.random.choice(np.arange(k + 1), p=weights[row] / weights[row].sum())

        rho = beta_distribution(self.split_beta, self.split_beta).rvs(n)
        rho = np.clip(np.asarray(rho, dtype=np.float64), 1e-12, 1.0 - 1e-12)

        new_weights = np.zeros((n, k + 2), dtype=np.float64)
        split_weight = weights[np.arange(n), split_idx]

        for row in range(n):
            j = int(split_idx[row])
            new_weights[row, :j] = weights[row, :j]
            new_weights[row, j] = rho[row] * weights[row, j]
            new_weights[row, j + 1] = (1.0 - rho[row]) * weights[row, j]
            new_weights[row, j + 2 :] = weights[row, j + 1 :]

        new_x = self._weights_to_raw(new_weights)
        prop_theta = self.setVariable(block_theta.copy(), self.indicator_name, np.full(n, int(new_mk[0])))
        prop_theta = self.deconcatParameters(new_x, prop_theta, new_mk)

        log_jac = (
            np.log(np.clip(split_weight, 1e-300, 1.0))
            + self._raw_to_weight_logdet(x)
            - self._raw_to_weight_logdet(new_x)
        )
        log_forward_select = self._split_selection_log_prob(weights, split_idx)
        log_forward_aux = beta_distribution(self.split_beta, self.split_beta).logpdf(rho)
        log_reverse_select = self._merge_selection_log_prob(int(new_mk[0]), n)
        logpqratio = log_reverse_select - log_forward_select - log_forward_aux + log_jac
        return prop_theta, logpqratio

    def _semantic_birth_learned(self, block_theta, mk, new_mk):
        k = int(mk[0])
        n = block_theta.shape[0]
        x = self.concatParameters(block_theta, mk)
        weights = self._raw_to_weights(x)

        split_idx = np.zeros(n, dtype=int)
        for row in range(n):
            split_idx[row] = np.random.choice(np.arange(k + 1), p=weights[row] / weights[row].sum())

        rho = np.zeros(n, dtype=np.float64)
        log_forward_aux = np.zeros(n, dtype=np.float64)
        for j in range(k + 1):
            idx = split_idx == j
            if not idx.any():
                continue
            mu, sigma = self._semantic_learned_logit_params(x[idx], k, j)
            logit_rho = mu + sigma * np.random.randn(idx.sum())
            logit_rho = np.asarray(logit_rho, dtype=np.float64).reshape(idx.sum())
            rho[idx] = self._sigmoid_np(logit_rho)
            log_forward_aux[idx] = norm(mu, sigma).logpdf(logit_rho) - np.log(
                np.clip(rho[idx] * (1.0 - rho[idx]), 1e-300, 1.0)
            )
        rho = np.clip(rho, 1e-12, 1.0 - 1e-12)

        new_weights = np.zeros((n, k + 2), dtype=np.float64)
        split_weight = weights[np.arange(n), split_idx]

        for row in range(n):
            j = int(split_idx[row])
            new_weights[row, :j] = weights[row, :j]
            new_weights[row, j] = rho[row] * weights[row, j]
            new_weights[row, j + 1] = (1.0 - rho[row]) * weights[row, j]
            new_weights[row, j + 2 :] = weights[row, j + 1 :]

        new_x = self._weights_to_raw(new_weights)
        prop_theta = self.setVariable(block_theta.copy(), self.indicator_name, np.full(n, int(new_mk[0])))
        prop_theta = self.deconcatParameters(new_x, prop_theta, new_mk)

        log_jac = (
            np.log(np.clip(split_weight, 1e-300, 1.0))
            + self._raw_to_weight_logdet(x)
            - self._raw_to_weight_logdet(new_x)
        )
        log_forward_select = self._split_selection_log_prob(weights, split_idx)
        log_reverse_select = self._merge_selection_log_prob(int(new_mk[0]), n)
        logpqratio = log_reverse_select - log_forward_select - log_forward_aux + log_jac
        return prop_theta, logpqratio

    def _semantic_death(self, block_theta, mk, new_mk):
        k = int(mk[0])
        n = block_theta.shape[0]
        x = self.concatParameters(block_theta, mk)
        weights = self._raw_to_weights(x)

        merge_idx = np.random.randint(k, size=n)
        new_weights = np.zeros((n, k), dtype=np.float64)
        merged_weight = np.zeros(n, dtype=np.float64)
        rho = np.zeros(n, dtype=np.float64)

        for row in range(n):
            j = int(merge_idx[row])
            merged_weight[row] = weights[row, j] + weights[row, j + 1]
            rho[row] = weights[row, j] / merged_weight[row]
            new_weights[row, :j] = weights[row, :j]
            new_weights[row, j] = merged_weight[row]
            new_weights[row, j + 1 :] = weights[row, j + 2 :]

        rho = np.clip(rho, 1e-12, 1.0 - 1e-12)
        new_x = self._weights_to_raw(new_weights)
        prop_theta = self.setVariable(block_theta.copy(), self.indicator_name, np.full(n, int(new_mk[0])))
        prop_theta = self.deconcatParameters(new_x, prop_theta, new_mk)

        log_birth_jac = (
            np.log(np.clip(merged_weight, 1e-300, 1.0))
            + self._raw_to_weight_logdet(new_x)
            - self._raw_to_weight_logdet(x)
        )
        log_forward_select = self._merge_selection_log_prob(k, n)
        log_reverse_select = self._split_selection_log_prob(new_weights, merge_idx)
        log_reverse_aux = beta_distribution(self.split_beta, self.split_beta).logpdf(rho)
        logpqratio = log_reverse_select + log_reverse_aux - log_forward_select - log_birth_jac
        return prop_theta, logpqratio

    def _semantic_death_learned(self, block_theta, mk, new_mk):
        k = int(mk[0])
        n = block_theta.shape[0]
        x = self.concatParameters(block_theta, mk)
        weights = self._raw_to_weights(x)

        merge_idx = np.random.randint(k, size=n)
        new_weights = np.zeros((n, k), dtype=np.float64)
        merged_weight = np.zeros(n, dtype=np.float64)
        rho = np.zeros(n, dtype=np.float64)

        for row in range(n):
            j = int(merge_idx[row])
            merged_weight[row] = weights[row, j] + weights[row, j + 1]
            rho[row] = weights[row, j] / merged_weight[row]
            new_weights[row, :j] = weights[row, :j]
            new_weights[row, j] = merged_weight[row]
            new_weights[row, j + 1 :] = weights[row, j + 2 :]

        rho = np.clip(rho, 1e-12, 1.0 - 1e-12)
        new_x = self._weights_to_raw(new_weights)
        prop_theta = self.setVariable(block_theta.copy(), self.indicator_name, np.full(n, int(new_mk[0])))
        prop_theta = self.deconcatParameters(new_x, prop_theta, new_mk)

        log_birth_jac = (
            np.log(np.clip(merged_weight, 1e-300, 1.0))
            + self._raw_to_weight_logdet(new_x)
            - self._raw_to_weight_logdet(x)
        )
        log_forward_select = self._merge_selection_log_prob(k, n)
        log_reverse_select = self._split_selection_log_prob(new_weights, merge_idx)

        log_reverse_aux = np.zeros(n, dtype=np.float64)
        for j in range(k):
            idx = merge_idx == j
            if not idx.any():
                continue
            log_reverse_aux[idx] = self._semantic_learned_logpdf(rho[idx], new_x[idx], k - 1, j)

        logpqratio = log_reverse_select + log_reverse_aux - log_forward_select - log_birth_jac
        return prop_theta, logpqratio

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
                if self.between_model_move in {"semantic", "semantic_learned"}:
                    jump_distribution = self._jump_distribution(k)
                    choices = np.array(list(jump_distribution.keys()), dtype=int)
                    probs = np.array(list(jump_distribution.values()), dtype=float)
                    proposed_k = np.random.choice(choices, p=probs, size=jump_idx.sum())

                    jump_logpq = np.zeros(jump_idx.sum())
                    jump_prop_theta = mk_theta[jump_idx].copy()

                    for new_k in np.unique(proposed_k):
                        local_idx = proposed_k == new_k
                        new_mk = (int(new_k),)
                        block_theta = mk_theta[jump_idx][local_idx].copy()

                        if new_k == k + 1:
                            if self.between_model_move == "semantic_learned":
                                prop_jump, local_logpq = self._semantic_birth_learned(block_theta, mk, new_mk)
                            else:
                                prop_jump, local_logpq = self._semantic_birth(block_theta, mk, new_mk)
                        elif new_k == k - 1:
                            if self.between_model_move == "semantic_learned":
                                prop_jump, local_logpq = self._semantic_death_learned(block_theta, mk, new_mk)
                            else:
                                prop_jump, local_logpq = self._semantic_death(block_theta, mk, new_mk)
                        else:
                            raise ValueError(f"Unsupported jump from {k} to {new_k}")

                        jump_prop_theta[local_idx] = prop_jump
                        jump_logpq[local_idx] = local_logpq + self._model_log_ratio(k, int(new_k))

                    prop_mk_theta[jump_idx] = jump_prop_theta
                    mk_logpq[jump_idx] = jump_logpq

                else:
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


class RJFlowGlobalChangePointProposalCNF(RJFlowGlobalChangePointProposalVINF):
    """Sample-trained conditional shared flow for change-point proposals."""

    def __init__(self, *, problem, indicator_name, segment_names, within_model_proposal, **kwargs):
        super().__init__(
            normalizing_flows=None,
            problem=problem,
            indicator_name=indicator_name,
            segment_names=segment_names,
            within_model_proposal=within_model_proposal,
            **kwargs,
        )

    def calibratemmmpd(self, mmmpd, size, t):
        within_start = time.perf_counter()
        self.within_model_proposal.calibratemmmpd(mmmpd, size, t)
        self.within_model_calibration_seconds = time.perf_counter() - within_start

        if not self.use_conditional_shared_flow:
            raise NotImplementedError(
                "RJFlowGlobalChangePointProposalCNF currently only supports the shared conditional flow setting."
            )

        td_start = time.perf_counter()
        self.flows = {}
        self.mk_logZhat = {}
        mklist = self.pmodel.getModelKeys()

        theta, theta_w = mmmpd.getOriginalParticleDensityForTemperature(t, resample=False)
        model_key_indices, _ = self.pmodel.enumerateModels(theta)

        x_aug_list = []
        context_list = []
        context_mask_list = []
        weight_list = []

        observed_models = [mk for mk in mklist if mk in model_key_indices]
        if not observed_models:
            raise RuntimeError("No model-specific calibration draws were available for CNF training.")

        per_model_mass = 1.0 / len(observed_models)
        for mk in observed_models:
            idx = model_key_indices[mk]
            mk_theta = theta[idx]
            mk_active = self.concatParameters(mk_theta, mk)
            k = int(mk[0])
            inactive_dim = self.problem.k_max - k

            if inactive_dim > 0:
                inactive = norm(0, self.aux_scale).rvs((mk_theta.shape[0], inactive_dim))
                inactive = np.asarray(inactive, dtype=np.float64).reshape(mk_theta.shape[0], inactive_dim)
                x_aug = np.column_stack([mk_active, inactive])
            else:
                x_aug = mk_active.copy()

            context = self._context_for_model(mk, mk_theta.shape[0], dtype=torch.float32).detach().cpu().numpy()
            context_mask = ~context.astype(bool)

            mk_log_w = theta_w[idx]
            mk_weights = np.exp(mk_log_w - logsumexp(mk_log_w)) * per_model_mass

            x_aug_list.append(np.asarray(x_aug, dtype=np.float32))
            context_list.append(np.asarray(context, dtype=np.float32))
            context_mask_list.append(context_mask)
            weight_list.append(np.asarray(mk_weights, dtype=np.float64))

        x_aug = np.vstack(x_aug_list)
        context_inputs = np.vstack(context_list)
        context_mask = np.vstack(context_mask_list)
        input_weights = np.concatenate(weight_list)
        input_weights = input_weights / input_weights.sum()
        x_aug_t = torch.tensor(x_aug, dtype=torch.float32)
        input_weights_t = torch.tensor(input_weights, dtype=torch.float32)

        folder = f"{self.__class__.__name__}_collapsed_ctp_cnf_v1"
        self.shared_flow = train_with_checkpoint(
            self.save_flows_dir,
            folder,
            ("shared", self.problem.k_max),
            ConditionalMaskedRationalQuadraticFlow.factory,
            x_aug,
            context_inputs,
            context_mask,
            aux_dist=torch.distributions.normal.Normal(0.0, self.aux_scale),
            base_dist=StandardNormal((self.problem.k_max,)),
            boxing_transform=CauchyCDF(),
            initial_transform=FixedNorm(x_aug_t, input_weights_t),
            input_weights=input_weights,
        )
        self.conditional_target = None

        for mk in mklist:
            self.mk_logZhat[mk] = -np.log(len(mklist))
            self.flows[mk] = self.shared_flow
        self.td_calibration_seconds = time.perf_counter() - td_start

    def _transform_to_base_matrix(self, inputs, mk):
        x_aug, inactive = self._pack_augmented_parameters(inputs, mk)
        context = self._context_for_model(mk, inputs.shape[0], dtype=torch.float32)
        xx, logdet = self.shared_flow._transform.inverse(
            torch.tensor(x_aug, dtype=torch.float32),
            context=context,
        )
        xx_np = np.asarray(xx.detach().cpu().tolist(), dtype=np.float64)
        logdet_np = np.asarray(logdet.detach().cpu().tolist(), dtype=np.float64)
        inactive_log_prob = self._reference_log_prob_np(inactive)
        return xx_np, logdet_np, inactive_log_prob

    def _transform_from_base_matrix(self, x, inputs, mk):
        context = self._context_for_model(mk, x.shape[0], dtype=torch.float32)
        xx, logdet = self.shared_flow._transform.forward(
            torch.tensor(x, dtype=torch.float32),
            context=context,
        )
        xx_np = np.asarray(xx.detach().cpu().tolist(), dtype=np.float64)
        logdet_np = np.asarray(logdet.detach().cpu().tolist(), dtype=np.float64)
        prop_theta, inactive = self._unpack_augmented_parameters(xx_np, inputs, mk)
        inactive_log_prob = self._reference_log_prob_np(inactive)
        return prop_theta, logdet_np, inactive_log_prob

    def sample_model_parameters_from_flow(self, mk, size):
        theta = np.zeros((size, self.pmodel.dim()), dtype=np.float64)
        theta = self.setVariable(theta, self.indicator_name, np.full(size, int(mk[0])))

        context = self._context_for_model(mk, size, dtype=torch.float32)
        base_noise = torch.randn((size, self.problem.k_max), dtype=torch.float32)
        samples, _ = self.shared_flow._transform.inverse(base_noise, context=context)
        x_aug = np.asarray(samples.detach().cpu().tolist(), dtype=np.float64)
        theta, _ = self._unpack_augmented_parameters(x_aug, theta, mk)
        return self.pmodel.sanitise(theta)
