import numpy as np
import torch
from scipy.stats import poisson
from torch.distributions import Dirichlet
from torch.distributions.transforms import StickBreakingTransform

from src.variables import ConditionalVariableBlock, ImproperRV, ParametricModelSpace, UniformIntegerRV


class ChangePointModel(ParametricModelSpace):
    def __init__(self, proposal_class, problem, **kwargs):
        self.proposal_class = proposal_class
        self.problem = problem

        self.k_min = 0
        self.k_max = int(self.problem.k_max)
        self.indicator_name = "k"
        self.segment_names = [f"s{i}" for i in range(self.k_max)]
        # h is analytically collapsed out of the RJMCMC state.
        self.rate_names = [f"h{i}" for i in range(self.k_max + 1)]

        self._targets = {}
        self._stick_break = StickBreakingTransform()
        self._truncation_log_norm = float(np.log(poisson.cdf(self.k_max, self.problem.lambda_)))

        self.random_variables = self._setup_random_variables()
        self.proposal = self._setup_proposal(**kwargs)

        super().__init__(self.random_variables, self.proposal)

    def _setup_random_variables(self):
        blockrv = {}
        conditions = {}

        for i, name in enumerate(self.segment_names):
            blockrv[name] = ImproperRV()
            conditions[name] = i + 1

        return {
            "cp": ConditionalVariableBlock(
                blockrv,
                conditions,
                UniformIntegerRV(self.k_min, self.k_max),
                self.indicator_name,
            )
        }

    def _setup_proposal(self, **kwargs):
        raise NotImplementedError

    def _get_target(self, k):
        if k not in self._targets:
            self._targets[k] = self.problem.target(k)
        return self._targets[k]

    def _concat_active_parameters(self, theta, mk):
        k = int(mk[0])
        cols = self.generateRVIndices(model_key=mk)
        parts = []

        for name in self.segment_names[:k]:
            if name in cols:
                parts.append(theta[:, cols[name]])

        if len(parts) == 0:
            return np.zeros((theta.shape[0], 0))

        return np.column_stack(parts)

    def _sample_model_prior(self, size):
        ks = np.arange(self.k_min, self.k_max + 1)
        log_probs = poisson.logpmf(ks, self.problem.lambda_) - self._truncation_log_norm
        probs = np.exp(log_probs - log_probs.max())
        probs = probs / probs.sum()
        return np.random.choice(ks, size=size, p=probs)

    def draw(self, size=1):
        theta = np.zeros((size, self.dim()))
        cols = self.generateRVIndices()
        k_col = cols[self.indicator_name][0]
        sampled_k = self._sample_model_prior(size)
        theta[:, k_col] = sampled_k

        for k in np.unique(sampled_k):
            idx = sampled_k == k
            n_k = idx.sum()
            if n_k == 0:
                continue

            if k > 0:
                simplex = Dirichlet(torch.full((k + 1,), 2.0)).sample((n_k,))
                raw_s = self._stick_break.inv(simplex)
                for i in range(k):
                    theta[idx, cols[self.segment_names[i]][0]] = np.asarray(raw_s[:, i].tolist(), dtype=np.float64)

        return theta

    def draw_perfect(self, M):
        return self.draw(M)

    def sanitise(self, inputs):
        outputs = inputs.copy()
        mkdict, _ = self.enumerateModels(outputs)

        for mk, idx in mkdict.items():
            active = set(self.generateRVIndices(model_key=mk).keys())
            for name in self.segment_names:
                if name not in active:
                    outputs[idx] = self.proposal.setVariable(outputs[idx], name, 0.0)

        return outputs

    def compute_prior(self, theta):
        theta = self.sanitise(theta)
        log_prior = np.full(theta.shape[0], -np.inf)
        mkdict, _ = self.enumerateModels(theta)

        for mk, idx in mkdict.items():
            k = int(mk[0])
            target = self._get_target(k)
            active_theta = self._concat_active_parameters(theta[idx], mk)
            active_theta_t = torch.tensor(active_theta, dtype=torch.float64)

            param_log_prior = np.asarray(target._decode_and_log_prior(active_theta_t)[0].detach().cpu().tolist())
            model_log_prior = poisson.logpmf(k, self.problem.lambda_) - self._truncation_log_norm
            log_prior[idx] = model_log_prior + param_log_prior

        return log_prior

    def compute_llh(self, theta):
        theta = self.sanitise(theta)
        log_like = np.full(theta.shape[0], -np.inf)
        mkdict, _ = self.enumerateModels(theta)

        for mk, idx in mkdict.items():
            k = int(mk[0])
            target = self._get_target(k)
            active_theta = self._concat_active_parameters(theta[idx], mk)
            active_theta_t = torch.tensor(active_theta, dtype=torch.float64)
            log_like[idx] = np.asarray(target._compute_collapsed_log_likelihood(active_theta_t).detach().cpu().tolist())

        return log_like
