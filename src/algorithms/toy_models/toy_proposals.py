import numpy as np
import torch
from nflows.distributions.normal import StandardNormal
from normflows.distributions.base import DiagGaussian

from src.flows import Flow, RationalQuadraticFlow2
from src.transforms import (
    CompositeTransform,
    FixedNorm,
    InverseTransform,
    LTransform,
    NaiveGaussianTransform,
    SAS2DTransform,
    Sigmoid,
    SinArcSinhTransform,
)
from src.algorithms.utils import train_with_checkpoint

from .toy_proposal import RJToyModelProposal


class RJToyModelProposalAF(RJToyModelProposal):
    def __init__(self, problem, **kwargs):
        super().__init__(problem, **kwargs)

    def _calibratemmmpd(self, mks, mmmpd, t):
        for mk in mks:
            mk_theta, _ = mmmpd.getParticleDensityForModelAndTemperature(mk, t, resample=True, resample_max_size=2000)

            if self.getModelInt(mk) == 0:
                bnorm = StandardNormal((1,))

            elif self.getModelInt(mk) == 1:
                bnorm = StandardNormal((2,))

            else:
                raise ValueError("self.getModelInt(mk) should be 0 or 1")

            X = self.extractModelConcatCols(self.concatParameters(mk_theta, mk), mk)

            fn = InverseTransform(NaiveGaussianTransform(torch.Tensor(X)))

            self.flows[mk] = Flow(fn, bnorm)

        if self.verbose:
            print("MK Z", self.mk_logZhat)
            print("MK Z", [np.exp(z) for i, z in self.mk_logZhat.items()])


class RJToyModelProposalNF(RJToyModelProposal):
    def __init__(self, problem, **kwargs):
        super().__init__(problem, **kwargs)

    def _calibratemmmpd(self, mks, mmmpd, t):
        for mk in mks:
            mk_theta, _ = mmmpd.getParticleDensityForModelAndTemperature(mk, t, resample=True, resample_max_size=2000)

            if self.getModelInt(mk) == 0:
                ls = Sigmoid()
                bnorm = StandardNormal((1,))

            elif self.getModelInt(mk) == 1:
                ls = Sigmoid()
                bnorm = StandardNormal((2,))

            else:
                raise ValueError("self.getModelInt(mk) should be 0 or 1")

            X = self.extractModelConcatCols(self.concatParameters(mk_theta, mk), mk)

            fn = FixedNorm(torch.Tensor(X))
            self.flows[mk] = RationalQuadraticFlow2.factory(X, bnorm, ls, fn)

        if self.verbose:
            print("MK Z", self.mk_logZhat)
            print("MK Z", [np.exp(z) for i, z in self.mk_logZhat.items()])


class RJToyModelProposalPerfect(RJToyModelProposal):
    def __init__(self, problem, **kwargs):
        super().__init__(problem, **kwargs)

    def _calibratemmmpd(self, mks, mmmpd, t):
        for mk in mks:
            if self.getModelInt(mk) == 0:
                tf = InverseTransform(SinArcSinhTransform(self.problem.ep[0], self.problem.dp[0]))
                bnorm = StandardNormal((1,))

            elif self.getModelInt(mk) == 1:
                tf = InverseTransform(
                    CompositeTransform(
                        [
                            LTransform(torch.linalg.cholesky(torch.tensor([[1.0, 0.99], [0.99, 1.0]]))),
                            SAS2DTransform(self.problem.ep[1:3], self.problem.dp[1:3]),
                        ]
                    )
                )
                bnorm = StandardNormal((2,))

            else:
                raise ValueError("self.getModelInt(mk) should be 0 or 1")

            self.flows[mk] = Flow(tf, bnorm)

        if self.verbose:
            print("MK Z", self.mk_logZhat)
            print("MK Z", [np.exp(z) for i, z in self.mk_logZhat.items()])


class RJToyModelProposalVINF(RJToyModelProposal):
    def __init__(self, normalizing_flows, problem, **kwargs):
        self.normalizing_flows = normalizing_flows
        super().__init__(problem, **kwargs)

    def _calibratemmmpd(self, mks, mmmpd, t):
        for mk in mks:
            p = self.problem.target(k=self.getModelInt(mk))
            q0 = DiagGaussian(p.ndim)
            folder = self.__class__.__name__
            self.flows[mk] = train_with_checkpoint(
                self.save_flows_dir,
                folder,
                mk,
                self.normalizing_flows,
                q0=q0,
                target=p,
            )

        if self.verbose:
            print("MK Z", self.mk_logZhat)
            print("MK Z", [np.exp(z) for i, z in self.mk_logZhat.items()])

    def transformToBase(self, inputs, mk):
        X = self.extractModelConcatCols(self.concatParameters(inputs, mk), mk)
        XX, logdet = self.flows[mk]._transform.inverse(torch.tensor(X, dtype=torch.float32))
        return self.deconcatParameters(
            self.returnModelConcatCols(XX.detach().numpy(), mk), inputs, mk
        ), logdet.detach().numpy()

    def transformFromBase(self, inputs, mk):
        X = self.extractModelConcatCols(self.concatParameters(inputs, mk), mk)
        XX, logdet = self.flows[mk]._transform.forward(torch.tensor(X, dtype=torch.float32))
        return self.deconcatParameters(
            self.returnModelConcatCols(XX.detach().numpy(), mk), inputs, mk
        ), logdet.detach().numpy()
