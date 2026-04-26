from scripts.sas.sas_base import N_SAMPLES, configure_flow_training, run_rjmcmc_algorithm_based_ablation, setup_argparse
from scripts.sas.sas_vinfs import (
    get_normalizing_flows_6_5,
    get_normalizing_flows_6_8,
    get_normalizing_flows_6_11,
    get_normalizing_flows_9_5,
    get_normalizing_flows_9_8,
    get_normalizing_flows_9_11,
    get_normalizing_flows_12_5,
    get_normalizing_flows_12_8,
    get_normalizing_flows_12_11,
)
from src.utils.parallel import algorithm_based_run_with_fixed_resources
from src.vi_nflows import resolve_flow_training_device

ablations = [
    {"algorithm_suffix": "6and5_", "normalizing_flows": get_normalizing_flows_6_5},
    {"algorithm_suffix": "6and8_", "normalizing_flows": get_normalizing_flows_6_8},
    {"algorithm_suffix": "6and11_", "normalizing_flows": get_normalizing_flows_6_11},
    {"algorithm_suffix": "9and5_", "normalizing_flows": get_normalizing_flows_9_5},
    {"algorithm_suffix": "9and8_", "normalizing_flows": get_normalizing_flows_9_8},
    {"algorithm_suffix": "9and11_", "normalizing_flows": get_normalizing_flows_9_11},
    {"algorithm_suffix": "12and5_", "normalizing_flows": get_normalizing_flows_12_5},
    {"algorithm_suffix": "12and8_", "normalizing_flows": get_normalizing_flows_12_8},
    {"algorithm_suffix": "12and11_", "normalizing_flows": get_normalizing_flows_12_11},
]

PROBLEMS = ["SAS"]
ALGORITHMS = ["ToyModelVINF"]


if __name__ == "__main__":
    args = setup_argparse()
    configure_flow_training(
        args.flow_device,
        flow_num_samples=args.flow_num_samples,
        flow_hidden_layer_size=args.flow_hidden_layer_size,
        flow_annealing=args.flow_annealing,
    )
    run_device = "cuda" if resolve_flow_training_device(args.flow_device).startswith("cuda") else "cpu"

    algorithm_based_run_with_fixed_resources(
        run_rjmcmc_algorithm_based_ablation,
        algorithm_size=len(ablations),
        cores_per_task=1,
        max_cpu_ratio=0.9,
        # for run
        problems=PROBLEMS,
        algorithms=ALGORITHMS * len(ablations),
        start=args.start,
        end=args.end,
        ablations=ablations,
        # rjmcmc
        n_samples=N_SAMPLES,
        device=run_device,
        save_flows_dir=args.save_flows_dir,
    )
