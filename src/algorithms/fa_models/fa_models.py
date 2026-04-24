from src.proposals import ModelEnumerateProposal, SystematicChoiceProposal

from .fa_model import FactorAnalysisModel
from .fa_proposals import (
    FARWProposal,
    RJFlowGlobalFactorAnalysisProposalAF,
    RJFlowGlobalFactorAnalysisProposalVINFImportanceSampling,
    RJFlowGlobalFactorAnalysisProposalNF,
    RJFlowGlobalFactorAnalysisProposalVINF,
    RJFlowGlobalFactorAnalysisProposalVINFRejectionFree,
    RJGlobalFactorAnalysisProposalLW,
)


class FactorAnalysisModelLW(FactorAnalysisModel):
    def __init__(self, *, problem, y_data, **kwargs):
        super().__init__(problem, y_data, **kwargs)

    def _setup_proposal(self, k_min, k_max, **kwargs):
        transformedrwprop = FARWProposal(
            problem=self.problem,
            betaii_names=self.betaii_names,
            betaij_names=self.betaij_names,
            lambda_names=self.lambda_names,
            **kwargs,
        )
        trwmep = ModelEnumerateProposal(subproposal=transformedrwprop)
        if k_min < k_max:
            rjp = RJGlobalFactorAnalysisProposalLW(
                problem=self.problem,
                indicator_name="k",
                betaii_names=self.betaii_names,
                betaij_names=self.betaij_names,
                lambda_names=self.lambda_names,
                within_model_proposal=trwmep,
                **kwargs,
            )

            proposal = SystematicChoiceProposal([rjp, trwmep])
        else:
            proposal = trwmep

        return proposal


class FactorAnalysisModelAF(FactorAnalysisModel):
    def __init__(self, *, problem, y_data, **kwargs):
        super().__init__(problem, y_data, **kwargs)

    def _setup_proposal(self, k_min, k_max, **kwargs):
        transformedrwprop = FARWProposal(
            problem=self.problem,
            betaii_names=self.betaii_names,
            betaij_names=self.betaij_names,
            lambda_names=self.lambda_names,
            **kwargs,
        )
        trwmep = ModelEnumerateProposal(subproposal=transformedrwprop)
        if k_min < k_max:
            rjp = RJFlowGlobalFactorAnalysisProposalAF(
                problem=self.problem,
                indicator_name="k",
                betaii_names=self.betaii_names,
                betaij_names=self.betaij_names,
                lambda_names=self.lambda_names,
                within_model_proposal=trwmep,
                **kwargs,
            )

            proposal = SystematicChoiceProposal([rjp, trwmep])
        else:
            proposal = trwmep

        return proposal


class FactorAnalysisModelNF(FactorAnalysisModel):
    def __init__(self, *, problem, y_data, **kwargs):
        super().__init__(problem, y_data, **kwargs)

    def _setup_proposal(self, k_min, k_max, **kwargs):
        transformedrwprop = FARWProposal(
            problem=self.problem,
            betaii_names=self.betaii_names,
            betaij_names=self.betaij_names,
            lambda_names=self.lambda_names,
            **kwargs,
        )
        trwmep = ModelEnumerateProposal(subproposal=transformedrwprop)
        if k_min < k_max:
            rjp = RJFlowGlobalFactorAnalysisProposalNF(
                problem=self.problem,
                indicator_name="k",
                betaii_names=self.betaii_names,
                betaij_names=self.betaij_names,
                lambda_names=self.lambda_names,
                within_model_proposal=trwmep,
                **kwargs,
            )

            proposal = SystematicChoiceProposal([rjp, trwmep])
        else:
            proposal = trwmep

        return proposal


class FactorAnalysisModelVINF(FactorAnalysisModel):
    def __init__(self, *, normalizing_flows, problem, y_data, **kwargs):
        self.normalizing_flows = normalizing_flows
        super().__init__(problem, y_data, **kwargs)

    def _setup_proposal(self, k_min, k_max, **kwargs):
        transformedrwprop = FARWProposal(
            problem=self.problem,
            betaii_names=self.betaii_names,
            betaij_names=self.betaij_names,
            lambda_names=self.lambda_names,
            **kwargs,
        )
        trwmep = ModelEnumerateProposal(subproposal=transformedrwprop)
        if k_min < k_max:
            rjp = RJFlowGlobalFactorAnalysisProposalVINF(
                normalizing_flows=self.normalizing_flows,
                problem=self.problem,
                indicator_name="k",
                betaii_names=self.betaii_names,
                betaij_names=self.betaij_names,
                lambda_names=self.lambda_names,
                within_model_proposal=trwmep,
                **kwargs,
            )

            proposal = SystematicChoiceProposal([rjp, trwmep])
        else:
            proposal = trwmep

        return proposal


class FactorAnalysisModelVINFRF(FactorAnalysisModel):
    def __init__(self, *, normalizing_flows, problem, y_data, **kwargs):
        self.normalizing_flows = normalizing_flows
        super().__init__(problem, y_data, **kwargs)

    def _setup_proposal(self, k_min, k_max, **kwargs):
        posterior_model_probabilities = kwargs.pop(
            "posterior_model_probabilities",
            {
                (1,): 0.88,
                (2,): 0.12,
            },
        )
        transformedrwprop = FARWProposal(
            problem=self.problem,
            betaii_names=self.betaii_names,
            betaij_names=self.betaij_names,
            lambda_names=self.lambda_names,
            **kwargs,
        )
        trwmep = ModelEnumerateProposal(subproposal=transformedrwprop)
        if k_min < k_max:
            rjp = RJFlowGlobalFactorAnalysisProposalVINFRejectionFree(
                normalizing_flows=self.normalizing_flows,
                problem=self.problem,
                indicator_name="k",
                betaii_names=self.betaii_names,
                betaij_names=self.betaij_names,
                lambda_names=self.lambda_names,
                within_model_proposal=trwmep,
                posterior_model_probabilities=posterior_model_probabilities,
                **kwargs,
            )

            proposal = SystematicChoiceProposal([rjp, trwmep])
        else:
            proposal = trwmep

        return proposal


class FactorAnalysisModelVINFIS(FactorAnalysisModel):
    def __init__(self, *, normalizing_flows, problem, y_data, **kwargs):
        self.normalizing_flows = normalizing_flows
        super().__init__(problem, y_data, **kwargs)

    def _setup_proposal(self, k_min, k_max, **kwargs):
        transformedrwprop = FARWProposal(
            problem=self.problem,
            betaii_names=self.betaii_names,
            betaij_names=self.betaij_names,
            lambda_names=self.lambda_names,
            **kwargs,
        )
        trwmep = ModelEnumerateProposal(subproposal=transformedrwprop)
        if k_min < k_max:
            rjp = RJFlowGlobalFactorAnalysisProposalVINFImportanceSampling(
                normalizing_flows=self.normalizing_flows,
                problem=self.problem,
                indicator_name="k",
                betaii_names=self.betaii_names,
                betaij_names=self.betaij_names,
                lambda_names=self.lambda_names,
                within_model_proposal=trwmep,
                **kwargs,
            )

            proposal = SystematicChoiceProposal([rjp, trwmep])
        else:
            proposal = trwmep

        return proposal
