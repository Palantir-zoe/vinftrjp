__all__ = [
    "ChangePointModelVINF",
    "FactorAnalysisModelAF",
    "FactorAnalysisModelLW",
    "FactorAnalysisModelNF",
    "FactorAnalysisModelVINF",
    "FactorAnalysisModelVINFRF",
    "ToyModelAF",
    "ToyModelNF",
    "ToyModelPerfect",
    "ToyModelVINF",
    "RobustBlockVSModelIndivSMC",
    "RobustBlockVSModelIndivAF",
    "RobustBlockVSModelIndivRQ",
    "RobustBlockVSModelCNF",
    "RobustBlockVSModelNaive",
    "RobustBlockVSModelIndivVINF",
    "RobustBlockVSModelVINF",
]

from .change_point_models import ChangePointModelVINF
from .fa_models import (
    FactorAnalysisModelAF,
    FactorAnalysisModelLW,
    FactorAnalysisModelNF,
    FactorAnalysisModelVINF,
    FactorAnalysisModelVINFRF,
)
from .toy_models import ToyModelAF, ToyModelNF, ToyModelPerfect, ToyModelVINF
from .vs_models import (
    RobustBlockVSModelCNF,
    RobustBlockVSModelIndivAF,
    RobustBlockVSModelIndivRQ,
    RobustBlockVSModelIndivSMC,
    RobustBlockVSModelIndivVINF,
    RobustBlockVSModelNaive,
    RobustBlockVSModelVINF,
)

ALGORITHMS = {
    "ChangePointModelVINF": ChangePointModelVINF,
    "FactorAnalysisModelAF": FactorAnalysisModelAF,
    "FactorAnalysisModelLW": FactorAnalysisModelLW,
    "FactorAnalysisModelNF": FactorAnalysisModelNF,
    "FactorAnalysisModelVINF": FactorAnalysisModelVINF,
    "FactorAnalysisModelVINFRF": FactorAnalysisModelVINFRF,
    "ToyModelAF": ToyModelAF,
    "ToyModelNF": ToyModelNF,
    "ToyModelPerfect": ToyModelPerfect,
    "ToyModelVINF": ToyModelVINF,
    "RobustBlockVSModelIndivSMC": RobustBlockVSModelIndivSMC,
    "RobustBlockVSModelIndivAF": RobustBlockVSModelIndivAF,
    "RobustBlockVSModelIndivRQ": RobustBlockVSModelIndivRQ,
    "RobustBlockVSModelCNF": RobustBlockVSModelCNF,
    "RobustBlockVSModelNaive": RobustBlockVSModelNaive,
    "RobustBlockVSModelIndivVINF": RobustBlockVSModelIndivVINF,
    "RobustBlockVSModelVINF": RobustBlockVSModelVINF,
}


def get_algorithm(name: str, *args, **kwargs):
    if name not in ALGORITHMS:
        raise Exception(f"Algorithm: {name} is not found.")

    return ALGORITHMS[name](*args, **kwargs)
