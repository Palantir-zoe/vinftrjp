from typing import Any

import numpy as np

from src.proposals import EigDecComponentwiseNormalProposalTrial, ModelEnumerateProposal, SystematicChoiceProposal
from src.variables import NormalRV, RandomVariableBlock, TransDimensionalBlock, UniformIntegerRV

from .vs_model import RobustBlockVSModel
from .vs_proposals import (
    RJZGlobalBlockVSProposalIndivAffine,
    RJZGlobalBlockVSProposalIndivRQ,
    RJZGlobalBlockVSProposalIndivVinf,
    RJZGlobalRobustBlockVSProposalSaturatedCNF,
    RJZGlobalRobustBlockVSProposalSaturatedNaive,
    RJZGlobalRobustBlockVSProposalSaturatedVinf,
)


class RobustBlockVSModelIndivSMC(RobustBlockVSModel):
    def __init__(self, problem, k, **kwargs):
        super().__init__(problem, k, rj_only=False, **kwargs)

    def _setup_random_variables(self, k, rj_only, **kwargs):
        random_variables = {}
        for i in range(self.nblocks):
            cbs = int(np.array(self.blocksizes[:i]).sum())
            block_rvs: dict[str, Any] = {self.betanames[cbs + j]: NormalRV(0, 10) for j in range(self.blocksizes[i])}
            block_rvs[self.gammanames[i]] = UniformIntegerRV(self.minblockcount[i], self.maxblockcount[i])
            random_variables[self.blocknames[i]] = RandomVariableBlock(block_rvs)

        return random_variables

    def _setup_proposal(self, k, rj_only, **kwargs):
        mep = ModelEnumerateProposal(subproposal=EigDecComponentwiseNormalProposalTrial(proposals=self.betanames))
        proposal = mep
        return proposal

    def getModelIdentifier(self):
        """Overrides the built-in method, returns the k variable."""
        ids = self.gammanames
        for name in self.rv_names:
            i = self.rv[name].getModelIdentifier()
            if i is not None:
                ids.append(i)  # TODO: i or id?
        if len(ids) == 0:
            return None
        else:
            return ids


class RobustBlockVSModelIndivAF(RobustBlockVSModel):
    def __init__(self, problem, **kwargs):
        super().__init__(problem, k=None, rj_only=True, **kwargs)

    def _setup_random_variables(self, k, rj_only, **kwargs):
        random_variables = {}
        for i in range(self.nblocks):
            cbs = int(np.array(self.blocksizes[:i]).sum())
            block_rvs = {self.betanames[cbs + j]: NormalRV(0, 10) for j in range(self.blocksizes[i])}
            random_variables[self.blocknames[i]] = TransDimensionalBlock(
                block_rvs,
                nblocks_name=self.gammanames[i],
                minimum_blocks=self.minblockcount[i],
                maximum_blocks=self.maxblockcount[i],
                nblocks_position="last",
            )

        return random_variables

    def _setup_proposal(self, k, rj_only, **kwargs):
        mep = ModelEnumerateProposal(subproposal=EigDecComponentwiseNormalProposalTrial(proposals=self.betanames))
        if k is None:
            if rj_only:
                proposal = RJZGlobalBlockVSProposalIndivAffine(
                    problem=self.problem,
                    blocksizes=self.blocksizes,
                    blocknames=self.blocknames,
                    gammanames=self.gammanames,
                    betanames=self.betanames,
                    within_model_proposal=mep,
                    **kwargs,
                )
            else:
                proposal = SystematicChoiceProposal(
                    [
                        RJZGlobalBlockVSProposalIndivAffine(
                            problem=self.problem,
                            blocksizes=self.blocksizes,
                            blocknames=self.blocknames,
                            gammanames=self.gammanames,
                            betanames=self.betanames,
                            within_model_proposal=mep,
                            **kwargs,
                        ),
                        mep,
                    ]
                )
        else:
            proposal = mep
        return proposal


class RobustBlockVSModelIndivRQ(RobustBlockVSModel):
    def __init__(self, problem, **kwargs):
        super().__init__(problem, k=None, rj_only=True, **kwargs)

    def _setup_random_variables(self, k, rj_only, **kwargs):
        random_variables = {}
        for i in range(self.nblocks):
            cbs = int(np.array(self.blocksizes[:i]).sum())
            block_rvs = {self.betanames[cbs + j]: NormalRV(0, 10) for j in range(self.blocksizes[i])}
            random_variables[self.blocknames[i]] = TransDimensionalBlock(
                block_rvs,
                nblocks_name=self.gammanames[i],
                minimum_blocks=self.minblockcount[i],
                maximum_blocks=self.maxblockcount[i],
                nblocks_position="last",
            )

        return random_variables

    def _setup_proposal(self, k, rj_only, **kwargs):
        mep = ModelEnumerateProposal(subproposal=EigDecComponentwiseNormalProposalTrial(proposals=self.betanames))
        if k is None:
            if rj_only:
                proposal = RJZGlobalBlockVSProposalIndivRQ(
                    problem=self.problem,
                    blocksizes=self.blocksizes,
                    blocknames=self.blocknames,
                    gammanames=self.gammanames,
                    betanames=self.betanames,
                    within_model_proposal=mep,
                    **kwargs,
                )
            else:
                proposal = SystematicChoiceProposal(
                    [
                        RJZGlobalBlockVSProposalIndivRQ(
                            problem=self.problem,
                            blocksizes=self.blocksizes,
                            blocknames=self.blocknames,
                            gammanames=self.gammanames,
                            betanames=self.betanames,
                            within_model_proposal=mep,
                            **kwargs,
                        ),
                        mep,
                    ]
                )
        else:
            proposal = mep
        return proposal


class RobustBlockVSModelIndivVINF(RobustBlockVSModel):
    def __init__(self, problem, normalizing_flows, **kwargs):
        self.normalizing_flows = normalizing_flows
        super().__init__(problem, k=None, rj_only=True, **kwargs)

    def _setup_random_variables(self, k, rj_only, **kwargs):
        random_variables = {}
        for i in range(self.nblocks):
            cbs = int(np.array(self.blocksizes[:i]).sum())
            block_rvs = {self.betanames[cbs + j]: NormalRV(0, 10) for j in range(self.blocksizes[i])}
            random_variables[self.blocknames[i]] = TransDimensionalBlock(
                block_rvs,
                nblocks_name=self.gammanames[i],
                minimum_blocks=self.minblockcount[i],
                maximum_blocks=self.maxblockcount[i],
                nblocks_position="last",
            )

        return random_variables

    def _setup_proposal(self, k, rj_only, **kwargs):
        mep = ModelEnumerateProposal(subproposal=EigDecComponentwiseNormalProposalTrial(proposals=self.betanames))
        if k is None:
            if rj_only:
                proposal = RJZGlobalBlockVSProposalIndivVinf(
                    normalizing_flows=self.normalizing_flows,
                    problem=self.problem,
                    blocksizes=self.blocksizes,
                    blocknames=self.blocknames,
                    gammanames=self.gammanames,
                    betanames=self.betanames,
                    within_model_proposal=mep,
                    **kwargs,
                )
            else:
                proposal = SystematicChoiceProposal(
                    [
                        RJZGlobalBlockVSProposalIndivVinf(
                            normalizing_flows=self.normalizing_flows,
                            problem=self.problem,
                            blocksizes=self.blocksizes,
                            blocknames=self.blocknames,
                            gammanames=self.gammanames,
                            betanames=self.betanames,
                            within_model_proposal=mep,
                            **kwargs,
                        ),
                        mep,
                    ]
                )
        else:
            proposal = mep
        return proposal


class RobustBlockVSModelCNF(RobustBlockVSModel):
    def __init__(self, problem, **kwargs):
        super().__init__(problem, k=None, rj_only=True, **kwargs)

    def _setup_random_variables(self, k, rj_only, **kwargs):
        random_variables = {}
        for i in range(self.nblocks):
            cbs = int(np.array(self.blocksizes[:i]).sum())
            block_rvs: dict[str, Any] = {self.betanames[cbs + j]: NormalRV(0, 10) for j in range(self.blocksizes[i])}
            block_rvs[self.gammanames[i]] = UniformIntegerRV(self.minblockcount[i], self.maxblockcount[i])
            random_variables[self.blocknames[i]] = RandomVariableBlock(block_rvs)

        return random_variables

    def _setup_proposal(self, k, rj_only, **kwargs):
        mep = ModelEnumerateProposal(subproposal=EigDecComponentwiseNormalProposalTrial(proposals=self.betanames))
        if k is None:
            if rj_only:
                proposal = RJZGlobalRobustBlockVSProposalSaturatedCNF(
                    problem=self.problem,
                    blocksizes=self.blocksizes,
                    blocknames=self.blocknames,
                    gammanames=self.gammanames,
                    betanames=self.betanames,
                    within_model_proposal=mep,
                    **kwargs,
                )
            else:
                proposal = SystematicChoiceProposal(
                    [
                        RJZGlobalRobustBlockVSProposalSaturatedCNF(
                            problem=self.problem,
                            blocksizes=self.blocksizes,
                            blocknames=self.blocknames,
                            gammanames=self.gammanames,
                            betanames=self.betanames,
                            within_model_proposal=mep,
                            **kwargs,
                        ),
                        mep,
                    ]
                )
        else:
            proposal = mep
        return proposal

    def getModelIdentifier(self):
        """Overrides the built-in method, returns the k variable."""
        ids = self.gammanames
        for name in self.rv_names:
            i = self.rv[name].getModelIdentifier()
            if i is not None:
                ids.append(i)  # TODO: i or id?

        if len(ids) == 0:
            return None
        else:
            return ids


class RobustBlockVSModelNaive(RobustBlockVSModel):
    def __init__(self, problem, **kwargs):
        super().__init__(problem, k=None, rj_only=True, **kwargs)

    def _setup_random_variables(self, k, rj_only, **kwargs):
        random_variables = {}
        for i in range(self.nblocks):
            cbs = int(np.array(self.blocksizes[:i]).sum())
            block_rvs: dict[str, Any] = {self.betanames[cbs + j]: NormalRV(0, 10) for j in range(self.blocksizes[i])}
            block_rvs[self.gammanames[i]] = UniformIntegerRV(self.minblockcount[i], self.maxblockcount[i])
            random_variables[self.blocknames[i]] = RandomVariableBlock(block_rvs)

        return random_variables

    def _setup_proposal(self, k, rj_only, **kwargs):
        mep = ModelEnumerateProposal(subproposal=EigDecComponentwiseNormalProposalTrial(proposals=self.betanames))
        if k is None:
            if rj_only:
                proposal = RJZGlobalRobustBlockVSProposalSaturatedNaive(
                    problem=self.problem,
                    blocksizes=self.blocksizes,
                    blocknames=self.blocknames,
                    gammanames=self.gammanames,
                    betanames=self.betanames,
                    within_model_proposal=mep,
                    **kwargs,
                )
            else:
                proposal = SystematicChoiceProposal(
                    [
                        RJZGlobalRobustBlockVSProposalSaturatedNaive(
                            problem=self.problem,
                            blocksizes=self.blocksizes,
                            blocknames=self.blocknames,
                            gammanames=self.gammanames,
                            betanames=self.betanames,
                            within_model_proposal=mep,
                            **kwargs,
                        ),
                        mep,
                    ]
                )
        else:
            proposal = mep
        return proposal

    def getModelIdentifier(self):
        """Overrides the built-in method, returns the k variable."""
        ids = self.gammanames
        for name in self.rv_names:
            i = self.rv[name].getModelIdentifier()
            if i is not None:
                ids.append(i)  # TODO: i or id?
        if len(ids) == 0:
            return None
        else:
            return ids


class RobustBlockVSModelVINF(RobustBlockVSModel):
    def __init__(self, problem, normalizing_flows, **kwargs):
        self.normalizing_flows = normalizing_flows
        super().__init__(problem, k=None, rj_only=True, **kwargs)

    def _setup_random_variables(self, k, rj_only, **kwargs):
        random_variables = {}
        for i in range(self.nblocks):
            cbs = int(np.array(self.blocksizes[:i]).sum())
            block_rvs: dict[str, Any] = {self.betanames[cbs + j]: NormalRV(0, 10) for j in range(self.blocksizes[i])}
            block_rvs[self.gammanames[i]] = UniformIntegerRV(self.minblockcount[i], self.maxblockcount[i])
            random_variables[self.blocknames[i]] = RandomVariableBlock(block_rvs)

        return random_variables

    def _setup_proposal(self, k, rj_only, **kwargs):
        mep = ModelEnumerateProposal(subproposal=EigDecComponentwiseNormalProposalTrial(proposals=self.betanames))
        if k is None:
            if rj_only:
                proposal = RJZGlobalRobustBlockVSProposalSaturatedVinf(
                    normalizing_flows=self.normalizing_flows,
                    problem=self.problem,
                    blocksizes=self.blocksizes,
                    blocknames=self.blocknames,
                    gammanames=self.gammanames,
                    betanames=self.betanames,
                    within_model_proposal=mep,
                    **kwargs,
                )
            else:
                proposal = SystematicChoiceProposal(
                    [
                        RJZGlobalRobustBlockVSProposalSaturatedVinf(
                            normalizing_flows=self.normalizing_flows,
                            problem=self.problem,
                            blocksizes=self.blocksizes,
                            blocknames=self.blocknames,
                            gammanames=self.gammanames,
                            betanames=self.betanames,
                            within_model_proposal=mep,
                            **kwargs,
                        ),
                        mep,
                    ]
                )
        else:
            proposal = mep
        return proposal

    def getModelIdentifier(self):
        """Overrides the built-in method, returns the k variable."""
        ids = self.gammanames
        for name in self.rv_names:
            i = self.rv[name].getModelIdentifier()
            if i is not None:
                ids.append(i)  # TODO: i or id?
        if len(ids) == 0:
            return None
        else:
            return ids
