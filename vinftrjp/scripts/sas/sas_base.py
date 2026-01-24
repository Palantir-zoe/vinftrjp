import argparse

from src.main import Experiments

N_SAMPLES = 10000


DEFAULT_DATA_DICT = {
    "SAS": {
        "ToyModelAF": {"title": "Affine TRJ", "color": "darkorange", "alpha": 1},
        "ToyModelNF": {"title": "RQMA TRJ", "color": "green", "alpha": 0.9},
        "ToyModelPerfect": {"title": "Perfect TRJ", "color": "darkmagenta", "alpha": 0.7},
        "ToyModelVINF": {"title": "VI with NFs", "color": "blue", "alpha": 0.8},
    },
}


def setup_argparse():
    parser = argparse.ArgumentParser(description="Process start and end parameters")
    parser.add_argument("--start", type=int, nargs="?", default=1, help="Start value (default: 1)")
    parser.add_argument("--end", type=int, nargs="?", default=3, help="End value (default: 3)")
    return parser.parse_args()


def run_rjmcmc_algorithm_based_ablation(
    task_config, *, problems, algorithms, start, end, ablations, n_samples, **kwargs
) -> None:
    if len(algorithms) != len(ablations):
        raise ValueError(f"The lenght of {algorithms} and {ablations} should be equal.")

    problem_id = task_config.problem_id  # None
    algorithm_id = task_config.algorithm_id  # is not None

    p_start, p_end = 0, len(problems)
    if problem_id is not None:
        p_start, p_end = problem_id, problem_id + 1
    _problems = problems[p_start:p_end]

    a_start, a_end = 0, len(algorithms)
    if algorithm_id is not None:
        a_start, a_end = algorithm_id, algorithm_id + 1
    _algorithms = algorithms[a_start:a_end]

    ablation_id: int = problem_id if problem_id is not None else algorithm_id

    config = {"start": start, "end": end}

    # Problem-specific experiment execution
    e = Experiments(problems=_problems, algorithms=_algorithms, **config)
    e.run(
        algorithm_suffix=ablations[ablation_id]["algorithm_suffix"],
        # rjmcmc
        calibrate_draws=None,
        start_theta=None,
        n_samples=n_samples,
        # algorithm
        normalizing_flows=ablations[ablation_id]["normalizing_flows"],
        **kwargs,
    )
