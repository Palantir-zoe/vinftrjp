from scripts.sas.sas_base import N_SAMPLES, setup_argparse
from scripts.sas.sas_vinfs import get_normalizing_flows
from src.main import Experiments

PROBLEMS = ["SAS"]
ALGORITHMS = ["ToyModelAF", "ToyModelNF", "ToyModelPerfect", "ToyModelVINF"]


if __name__ == "__main__":
    args = setup_argparse()

    e = Experiments(args.start, args.end, problems=PROBLEMS, algorithms=ALGORITHMS)
    e.run(n_samples=N_SAMPLES, normalizing_flows=get_normalizing_flows)
