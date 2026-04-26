import argparse
import os

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


def _parse_optional_bool(value):
    if value is None:
        return None

    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def setup_argparse():
    parser = argparse.ArgumentParser(description="Process start and end parameters")
    parser.add_argument("--start", type=int, nargs="?", default=1, help="Start value (default: 1)")
    parser.add_argument("--end", type=int, nargs="?", default=3, help="End value (default: 3)")
    parser.add_argument(
        "--flow-device",
        type=str,
        default="cpu",
        help="Device for flow training only, e.g. 'cpu', 'cuda', or 'auto'.",
    )
    parser.add_argument(
        "--flow-num-samples",
        type=int,
        default=None,
        help="Override the number of Monte Carlo samples per flow-training iteration.",
    )
    parser.add_argument(
        "--flow-hidden-layer-size",
        type=int,
        default=None,
        help="Override the hidden width used by SAS flow training networks.",
    )
    parser.add_argument(
        "--flow-annealing",
        type=_parse_optional_bool,
        default=None,
        help="Override whether beta annealing is used during flow training (true/false).",
    )
    parser.add_argument(
        "--save-flows-dir",
        type=str,
        default="",
        help="Optional directory used to cache trained flows for reuse across runs.",
    )
    return parser.parse_args()


def configure_flow_training(
    flow_device: str,
    flow_num_samples: int | None = None,
    flow_hidden_layer_size: int | None = None,
    flow_annealing: bool | None = None,
) -> None:
    os.environ["FLOW_TRAIN_DEVICE"] = str(flow_device)

    if flow_num_samples is None:
        os.environ.pop("FLOW_TRAIN_NUM_SAMPLES", None)
    else:
        os.environ["FLOW_TRAIN_NUM_SAMPLES"] = str(flow_num_samples)

    if flow_hidden_layer_size is None:
        os.environ.pop("FLOW_HIDDEN_LAYER_SIZE", None)
    else:
        os.environ["FLOW_HIDDEN_LAYER_SIZE"] = str(flow_hidden_layer_size)

    if flow_annealing is None:
        os.environ.pop("FLOW_TRAIN_ANNEALING", None)
    else:
        os.environ["FLOW_TRAIN_ANNEALING"] = "true" if flow_annealing else "false"


def run_rjmcmc_algorithm_based_ablation(
    task_config, *, problems, algorithms, start, end, ablations, n_samples, device="cpu", **kwargs
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
    e = Experiments(problems=_problems, algorithms=_algorithms, device=device, **config)
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
