from src.proposals import (
    EigDecComponentwiseNormalProposalTrial,
    ModelEnumerateProposal,
    RWProposal,
    SystematicChoiceProposal,
)

from .toy_model import ToyModel
from .toy_proposals import RJToyModelProposalAF, RJToyModelProposalNF, RJToyModelProposalPerfect, RJToyModelProposalVINF


class ToyModelAF(ToyModel):
    def __init__(self, *, problem, **kwargs):
        super().__init__(RJToyModelProposalAF, problem, **kwargs)

    def _setup_proposal(self, **kwargs):
        wmp = ModelEnumerateProposal(subproposal=EigDecComponentwiseNormalProposalTrial(proposals=["t1", "t2"]))
        proposal = self.proposal_class(problem=self.problem, within_model_proposal=wmp, **kwargs)

        return proposal


class ToyModelNF(ToyModel):
    def __init__(self, *, problem, **kwargs):
        super().__init__(RJToyModelProposalNF, problem, **kwargs)

    def _setup_proposal(self, **kwargs):
        if self.run_index is None:
            wmp = ModelEnumerateProposal(subproposal=RWProposal(["t1", "t2"]))
            proposal = SystematicChoiceProposal(
                [
                    self.proposal_class(problem=self.problem, within_model_proposal=wmp, **kwargs),
                    wmp,
                ]
            )
        else:
            wmp = ModelEnumerateProposal(subproposal=EigDecComponentwiseNormalProposalTrial(proposals=["t1", "t2"]))
            proposal = self.proposal_class(problem=self.problem, within_model_proposal=wmp, **kwargs)

        return proposal


class ToyModelPerfect(ToyModel):
    def __init__(self, *, problem, **kwargs):
        super().__init__(RJToyModelProposalPerfect, problem, **kwargs)

    def _setup_proposal(self, **kwargs):
        if self.run_index is None:
            wmp = ModelEnumerateProposal(subproposal=RWProposal(["t1", "t2"]))
            proposal = SystematicChoiceProposal(
                [
                    self.proposal_class(problem=self.problem, within_model_proposal=wmp, **kwargs),
                    wmp,
                ]
            )
        else:
            wmp = ModelEnumerateProposal(subproposal=EigDecComponentwiseNormalProposalTrial(proposals=["t1", "t2"]))
            proposal = self.proposal_class(problem=self.problem, within_model_proposal=wmp, **kwargs)

        return proposal


class ToyModelVINF(ToyModel):
    def __init__(self, *, normalizing_flows, problem, **kwargs):
        self.normalizing_flows = normalizing_flows
        super().__init__(RJToyModelProposalVINF, problem, **kwargs)

    def _setup_proposal(self, **kwargs):
        if self.run_index is None:
            wmp = ModelEnumerateProposal(subproposal=RWProposal(["t1", "t2"]))
            proposal = SystematicChoiceProposal(
                [
                    self.proposal_class(
                        normalizing_flows=self.normalizing_flows,
                        problem=self.problem,
                        within_model_proposal=wmp,
                        **kwargs,
                    ),
                    wmp,
                ]
            )
        else:
            wmp = ModelEnumerateProposal(subproposal=EigDecComponentwiseNormalProposalTrial(proposals=["t1", "t2"]))
            proposal = self.proposal_class(
                normalizing_flows=self.normalizing_flows, problem=self.problem, within_model_proposal=wmp, **kwargs
            )

        return proposal
