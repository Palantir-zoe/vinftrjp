from src.proposals import ModelEnumerateProposal

from .change_point_model import ChangePointModel
from .change_point_proposal import (
    ChangePointWithinModelProposal,
    RJFlowGlobalChangePointProposalCNF,
    RJFlowGlobalChangePointProposalVINF,
)


class ChangePointModelVINF(ChangePointModel):
    def __init__(self, *, normalizing_flows, problem, **kwargs):
        self.normalizing_flows = normalizing_flows
        super().__init__(RJFlowGlobalChangePointProposalVINF, problem, **kwargs)

    def _setup_proposal(self, **kwargs):
        within_model_proposal = ModelEnumerateProposal(
            subproposal=ChangePointWithinModelProposal(
                problem=self.problem,
                segment_names=self.segment_names,
                proposal_scale=kwargs.get("within_model_scale", 0.35),
            )
        )
        return self.proposal_class(
            normalizing_flows=self.normalizing_flows,
            problem=self.problem,
            indicator_name=self.indicator_name,
            segment_names=self.segment_names,
            within_model_proposal=within_model_proposal,
            **kwargs,
        )


class ChangePointModelCNF(ChangePointModel):
    def __init__(self, *, problem, **kwargs):
        super().__init__(RJFlowGlobalChangePointProposalCNF, problem, **kwargs)

    def _setup_proposal(self, **kwargs):
        within_model_proposal = ModelEnumerateProposal(
            subproposal=ChangePointWithinModelProposal(
                problem=self.problem,
                segment_names=self.segment_names,
                proposal_scale=kwargs.get("within_model_scale", 0.35),
            )
        )
        return self.proposal_class(
            problem=self.problem,
            indicator_name=self.indicator_name,
            segment_names=self.segment_names,
            within_model_proposal=within_model_proposal,
            **kwargs,
        )


class ChangePointModelSMC(ChangePointModel):
    """Fixed-k change-point model used to generate posterior samples via SMC."""

    def __init__(self, *, problem, fixed_k, **kwargs):
        super().__init__(proposal_class=None, problem=problem, fixed_k=fixed_k, **kwargs)

    def _setup_proposal(self, **kwargs):
        return ModelEnumerateProposal(
            subproposal=ChangePointWithinModelProposal(
                problem=self.problem,
                segment_names=self.segment_names,
                proposal_scale=kwargs.get("within_model_scale", 0.35),
            )
        )
