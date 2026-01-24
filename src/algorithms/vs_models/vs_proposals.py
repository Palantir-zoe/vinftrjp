import numpy as np
import torch
import torch.nn as nn
from nflows.distributions.normal import ConditionalDiagonalNormal, StandardNormal
from normflows.distributions.base import ConditionalDiagGaussian, DiagGaussian
from scipy.special import logsumexp
from scipy.stats import norm

from src.flows import ConditionalMaskedRationalQuadraticFlow, Flow, RationalQuadraticFlow2
from src.transforms import (
    CompositeTransform,
    ConditionalMaskedTransform,
    FixedLinear,
    FixedNorm,
    IdentityTransform,
    InverseTransform,
    MaskedFixedNorm,
    NaiveGaussianTransform,
    Sigmoid,
)

from ..utils import train_with_checkpoint
from .vs_proposal import RJZGlobalBlockVSProposalIndiv, RJZGlobalRobustBlockVSProposalSaturated


class RJZGlobalBlockVSProposalIndivAffine(RJZGlobalBlockVSProposalIndiv):
    def __init__(self, problem, **kwargs):
        super().__init__(problem, **kwargs)

    def _make_flow(self, mk, mk_theta_w, ls, X, bnorm):
        # Use affine Gaussian transformation
        fn = InverseTransform(NaiveGaussianTransform(torch.Tensor(X), torch.Tensor(mk_theta_w)))

        folder = self.__class__.__name__
        flow = train_with_checkpoint(self.save_flows_dir, folder, mk, Flow, fn, bnorm)
        return flow


class RJZGlobalBlockVSProposalIndivRQ(RJZGlobalBlockVSProposalIndiv):
    def __init__(self, problem, **kwargs):
        super().__init__(problem, **kwargs)

    def _make_flow(self, mk, mk_theta_w, ls, X, bnorm):
        # Use rational quadratic flow
        fn = FixedNorm(torch.Tensor(X))

        folder = self.__class__.__name__
        flow = train_with_checkpoint(self.save_flows_dir, folder, mk, RationalQuadraticFlow2.factory, X, bnorm, ls, fn)
        return flow


class RJZGlobalBlockVSProposalIndivVinf(RJZGlobalBlockVSProposalIndivAffine):
    def __init__(self, normalizing_flows, problem, **kwargs):
        self.normalizing_flows = normalizing_flows
        super().__init__(problem, **kwargs)

    def _make_flow(self, mk, mk_theta_w, ls, X, bnorm):
        p = self.problem.target(k=list(mk))
        q0 = DiagGaussian(p.ndim)

        folder = self.__class__.__name__
        flow = train_with_checkpoint(self.save_flows_dir, folder, mk, self.normalizing_flows, q0=q0, target=p)
        return flow

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
        XX, logdet = self.flows[mk]._transform.inverse(torch.tensor(X, dtype=torch.float32))
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
        XX, logdet = self.flows[mk]._transform.forward(torch.tensor(X, dtype=torch.float32))
        return (
            self.deconcatParameters(XX.detach().numpy(), inputs, mk),
            logdet.detach().numpy(),
        )


class RJZGlobalRobustBlockVSProposalSaturatedNaive(RJZGlobalRobustBlockVSProposalSaturated):
    def __init__(self, problem, **kwargs):
        super().__init__(problem=problem, **kwargs)

    def _make_flow(self, mklist, mmmpd, t):
        # Initialize naive flow with identity transform
        theta, theta_w = mmmpd.getParticleDensityForTemperature(t, resample=False)
        rvidx = self.rv_indices[mklist[0]]
        X = np.column_stack([theta[:, rvidx[rvn]] for rvn in self.betanames])
        beta_dim = X.shape[1]

        folder = self.__class__.__name__
        flow = train_with_checkpoint(
            self.save_flows_dir, folder, mklist, Flow, IdentityTransform(), StandardNormal((beta_dim,))
        )
        return flow


class RJZGlobalRobustBlockVSProposalSaturatedCNF(RJZGlobalRobustBlockVSProposalSaturated):
    def __init__(self, problem, **kwargs):
        super().__init__(problem=problem, **kwargs)

    def _make_flow(self, mklist, mmmpd, t):
        # Train conditional normalizing flow on beta parameters
        theta, theta_w = mmmpd.getParticleDensityForTemperature(t, resample=False)
        rvidx = self.rv_indices[mklist[0]]
        X = np.column_stack([theta[:, rvidx[rvn]] for rvn in self.betanames])
        Y = np.column_stack([theta[:, rvidx[rvn]] for rvn in self.gammanames])

        # Initialize inactive parameters with prior samples
        U = norm(0, 10).rvs(X.shape)
        Ymask = Y.repeat(self.blocksizes, axis=1).astype(bool)
        X[~Ymask] = U[~Ymask]

        # Reweight to balance model representation for training
        un, un_inv = np.unique(Y, return_inverse=True, axis=0)
        for i, yu in enumerate(un):
            yidx = i == un_inv
            theta_w[yidx] = theta_w[yidx] - logsumexp(theta_w[yidx])
        theta_w = np.exp(theta_w - logsumexp(theta_w))

        beta_dim = X.shape[1]
        gamma_dim = Y.shape[1]

        # Build context network for conditional flow
        ls = Sigmoid()
        if True:
            args = [nn.Linear(gamma_dim, 128), nn.ReLU()]
            for i in range(29):
                args += [nn.Linear(128, 128), nn.ReLU()]
            args += [nn.Linear(128, 2 * beta_dim)]
            context_net = nn.Sequential(*args)

        bnorm = ConditionalDiagonalNormal(shape=(beta_dim,), context_encoder=context_net)

        # Create masking transforms for active/inactive parameters
        bs = torch.Tensor(tuple(self.blocksizes)).type(torch.int)
        fn_param = MaskedFixedNorm(
            torch.Tensor(X),
            torch.Tensor(theta_w),
            Ymask,
            lambda y: y.repeat_interleave(bs, dim=1).type(torch.bool),
        )
        fn_aux = ConditionalMaskedTransform(
            FixedLinear(shift=0, scale=1.0 / 10),
            lambda y: ~y.repeat_interleave(bs, dim=1).type(torch.bool),
        )
        fn = CompositeTransform([fn_param, fn_aux])

        # Initialize conditional rational quadratic flow
        priordist = torch.distributions.normal.Normal(0, 10)

        folder = self.__class__.__name__
        flow = train_with_checkpoint(
            self.save_flows_dir,
            folder,
            mklist,
            ConditionalMaskedRationalQuadraticFlow.factory,
            X,
            Y,
            ~Ymask,
            priordist,
            base_dist=bnorm,
            boxing_transform=ls,
            initial_transform=fn,
            input_weights=theta_w,
        )
        return flow


class RJZGlobalRobustBlockVSProposalSaturatedVinf(RJZGlobalRobustBlockVSProposalSaturated):
    def __init__(self, normalizing_flows, problem, **kwargs):
        self.normalizing_flows = normalizing_flows
        super().__init__(problem=problem, **kwargs)

    def _make_flow(self, mklist, mmmpd, t):
        p = self.problem.target()

        gamma_dim = 3
        beta_dim = 4  # p.ndim
        args = [nn.Linear(gamma_dim, 128), nn.ReLU()]
        for _ in range(29):
            args += [nn.Linear(128, 128), nn.ReLU()]
        args += [nn.Linear(128, 2 * beta_dim)]
        context_net = nn.Sequential(*args)

        q0 = ConditionalDiagGaussian(beta_dim, context_encoder=context_net)

        folder = self.__class__.__name__
        flow = train_with_checkpoint(self.save_flows_dir, folder, mklist, self.normalizing_flows, q0=q0, target=p)
        return flow

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
        XX, logdet = self.flow._transform.inverse(
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
        XX, logdet = self.flow._transform.forward(
            torch.tensor(X, dtype=torch.float32),
            context=torch.tensor(Y, dtype=torch.float32),
        )
        return (
            self.deconcatParameters(XX.detach().numpy(), inputs, mk),
            logdet.detach().numpy(),
        )
