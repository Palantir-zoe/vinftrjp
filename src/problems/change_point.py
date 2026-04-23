from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .problem import Problem


class ChangePoint(Problem):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.k_max = kwargs.get("k_max", 10)
        self.lambda_ = kwargs.get("lambda_", 3.0)
        self.alpha = kwargs.get("alpha", 1.0)
        self.beta = kwargs.get("beta", 200.0)

        self.event_times, self.L = self._load_data()
        self.ndim = None

    def _load_data(self):
        root = Path(__file__).resolve().parents[2]
        data = np.loadtxt(root / "change_point" / "Data_change-point.txt")

        # Match the R implementation: flatten by columns, then convert the inter-arrival
        # times to absolute event times and account for the leading/trailing zero-incident windows.
        intervals = np.asarray(data, dtype=np.float64).reshape(-1, order="F")
        event_times = np.cumsum(intervals) + 31 + 28 + 14
        horizon = float(event_times[-1] + 10 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30 + 31)

        return torch.tensor(event_times, dtype=torch.float32), horizon

    def target(self, k: int, **kwargs):
        target_ = ChangePointTarget(
            event_times=self.event_times,
            L=self.L,
            k=k,
            alpha=self.alpha,
            beta=self.beta,
        )
        self.ndim = target_.ndim
        return target_

    def conditional_target(self, **kwargs):
        target_ = ChangePointPaddedTarget(
            event_times=self.event_times,
            L=self.L,
            k_max=self.k_max,
            alpha=self.alpha,
            beta=self.beta,
            reference_scale=kwargs.get("reference_scale", 1.0),
        )
        self.ndim = target_.ndim
        return target_


class ChangePointTarget(nn.Module):
    def __init__(self, *, event_times, L, k, alpha, beta):
        super().__init__()

        self.event_times = event_times
        self.L = float(L)
        self.k = int(k)
        self.alpha = float(alpha)
        self.beta = float(beta)
        # Collapsed target: h is integrated analytically, so the flow only
        # learns the k unconstrained spacing/location coordinates.
        self.ndim = self.k

    def log_prob(self, theta):
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)

        log_prior, _, _ = self._decode_and_log_prior(theta)
        log_llh = self._compute_collapsed_log_likelihood(theta)
        return log_prior + log_llh

    def _stick_breaking_log_weights(self, raw_s):
        offset = (raw_s.shape[-1] + 1) - raw_s.new_ones(raw_s.shape[-1]).cumsum(-1)
        logits = raw_s - offset.log()
        log_z = F.logsigmoid(logits)
        log_one_minus_z = F.logsigmoid(-logits)

        prefix = torch.cumsum(log_one_minus_z, dim=-1)
        prev_prefix = torch.cat(
            [torch.zeros((raw_s.shape[0], 1), dtype=raw_s.dtype, device=raw_s.device), prefix[:, :-1]],
            dim=-1,
        )
        log_weights = torch.cat([prev_prefix + log_z, prefix[:, -1:].clone()], dim=-1)
        return log_weights, logits

    def _dirichlet_log_prob_from_log_weights(self, log_weights, alpha_vec):
        log_norm = torch.lgamma(alpha_vec.sum()) - torch.lgamma(alpha_vec).sum()
        return log_norm + ((alpha_vec - 1.0) * log_weights).sum(dim=-1)

    def _decode_and_log_prior(self, theta):
        theta = theta.to(dtype=torch.float64)
        raw_s = theta[:, : self.k]

        batch_size = theta.shape[0]
        device = theta.device
        dtype = theta.dtype
        valid = torch.isfinite(theta).all(dim=1)
        neg_inf = torch.full((batch_size,), -torch.inf, dtype=dtype, device=device)

        if self.k == 0:
            weights = torch.ones((batch_size, 1), dtype=dtype, device=device)
            log_det_s = torch.zeros(batch_size, dtype=dtype, device=device)
            change_points = torch.zeros((batch_size, 0), dtype=dtype, device=device)
            log_prior_s = torch.zeros(batch_size, dtype=dtype, device=device)
        else:
            safe_raw_s = torch.where(valid.unsqueeze(1), raw_s, torch.zeros_like(raw_s))
            log_weights, logits = self._stick_breaking_log_weights(safe_raw_s)
            weights = torch.softmax(log_weights, dim=-1)
            change_points = self.L * torch.cumsum(weights[:, :-1], dim=1)
            alpha_vec = torch.full((self.k + 1,), 2.0, dtype=dtype, device=device)
            log_prior_s = self._dirichlet_log_prob_from_log_weights(log_weights, alpha_vec)
            log_det_s = (-logits + F.logsigmoid(logits) + log_weights[:, :-1]).sum(dim=1)

        log_prior = log_prior_s + log_det_s
        valid = valid & torch.isfinite(log_prior) & torch.isfinite(weights).all(dim=1)
        log_prior = torch.where(valid, log_prior, neg_inf)

        return log_prior, weights, change_points

    def _segment_counts(self, segment_lengths, dtype, device):
        event_times = self.event_times.to(device=device, dtype=dtype)
        counts = []

        for i in range(segment_lengths.shape[0]):
            boundaries = torch.cat(
                [
                    torch.zeros(1, dtype=dtype, device=device),
                    torch.cumsum(segment_lengths[i], dim=0),
                ]
            )
            boundaries[-1] = torch.as_tensor(self.L, dtype=dtype, device=device)
            idx = torch.searchsorted(event_times, boundaries, right=True)
            counts.append(idx[1:] - idx[:-1])

        return torch.stack(counts, dim=0).to(dtype=dtype)

    def _compute_collapsed_log_likelihood(self, theta):
        log_prior, weights, _ = self._decode_and_log_prior(theta)
        segment_lengths = self.L * weights
        dtype = weights.dtype
        device = weights.device

        log_like = torch.full((theta.shape[0],), -torch.inf, dtype=dtype, device=device)
        valid = torch.isfinite(log_prior)
        if not valid.any():
            return log_like

        counts = self._segment_counts(segment_lengths, dtype, device)
        alpha = torch.as_tensor(self.alpha, dtype=dtype, device=device)
        beta = torch.as_tensor(self.beta, dtype=dtype, device=device)

        collapsed_terms = (
            alpha * torch.log(beta)
            - torch.lgamma(alpha)
            + torch.lgamma(alpha + counts)
            - (alpha + counts) * torch.log(beta + segment_lengths)
        )
        log_like[valid] = collapsed_terms.sum(dim=1)[valid]

        return log_like

    def posterior_rate_mean(self, theta):
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)

        theta = theta.to(dtype=torch.float64)
        _, weights, _ = self._decode_and_log_prior(theta)
        segment_lengths = self.L * weights
        counts = self._segment_counts(segment_lengths, theta.dtype, theta.device)
        return (self.alpha + counts) / (self.beta + segment_lengths)

    # Backwards-compatible aliases used by a few scripts.
    def _compute_log_likelihood(self, theta):
        return self._compute_collapsed_log_likelihood(theta)


class ChangePointPaddedTarget(nn.Module):
    def __init__(self, *, event_times, L, k_max, alpha, beta, reference_scale=1.0):
        super().__init__()

        self.event_times = event_times
        self.L = float(L)
        self.k_max = int(k_max)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.reference_scale = float(reference_scale)

        self.ndim = self.k_max
        self.context_dim = self.k_max
        self._targets = {}

    def _target(self, k):
        if k not in self._targets:
            self._targets[k] = ChangePointTarget(
                event_times=self.event_times,
                L=self.L,
                k=k,
                alpha=self.alpha,
                beta=self.beta,
            )
        return self._targets[k]

    def sample_context(self, batch_size, device=None):
        if device is None:
            device = self.event_times.device
        k = torch.randint(0, self.k_max + 1, (batch_size,), device=device)
        positions = torch.arange(self.k_max, device=device).unsqueeze(0)
        return (positions < k.unsqueeze(1)).to(dtype=torch.float32)

    def context_for_model(self, k, batch_size, device=None, dtype=torch.float32):
        if device is None:
            device = self.event_times.device
        context = torch.zeros((batch_size, self.context_dim), dtype=dtype, device=device)
        if int(k) > 0:
            context[:, : int(k)] = 1.0
        return context

    def _context_to_k(self, context):
        if context is None:
            raise ValueError("ChangePointPaddedTarget requires a model-index context.")
        if context.shape[1] != self.context_dim:
            raise ValueError(f"context must have shape (n, {self.context_dim}), got {tuple(context.shape)}")
        return torch.round(context.sum(dim=1)).to(dtype=torch.long)

    def _reference_log_prob(self, x):
        if x.shape[1] == 0:
            return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)

        scale = torch.as_tensor(self.reference_scale, dtype=x.dtype, device=x.device)
        return (-torch.log(scale) - 0.5 * np.log(2.0 * np.pi) - 0.5 * (x / scale) ** 2).sum(dim=1)

    def log_prob(self, theta, context=None):
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)

        theta = theta.to(dtype=torch.float64)
        context = context.to(device=theta.device)
        ks = self._context_to_k(context)
        logp = torch.full((theta.shape[0],), -torch.inf, dtype=theta.dtype, device=theta.device)

        for k_value in torch.unique(ks).detach().cpu().tolist():
            k = int(k_value)
            idx = ks == k
            active = theta[idx, :k]
            inactive = theta[idx, k:]
            logp[idx] = self._target(k).log_prob(active) + self._reference_log_prob(inactive)

        return logp
