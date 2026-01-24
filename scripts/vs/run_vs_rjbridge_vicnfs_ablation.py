from pathlib import Path

from scripts.vs.vs_base import (
    BASE_ALGORITHM,
    BLOCK_SIZE,
    K_LIST,
    NPARTICLES,
    get_train_theta_test_theta,
    run_rjbridge_algorithm_based_ablation,
    setup_argparse,
)
from scripts.vs.vs_vicnfs import (
    get_normalizing_flows_2,
    get_normalizing_flows_4,
    get_normalizing_flows_6,
    get_normalizing_flows_8,
    get_normalizing_flows_10,
    get_normalizing_flows_12,
    get_normalizing_flows_14,
    get_normalizing_flows_16,
    get_normalizing_flows_18,
    get_normalizing_flows_20,
    get_normalizing_flows_22,
    get_normalizing_flows_24,
    get_normalizing_flows_26,
    get_normalizing_flows_28,
    get_normalizing_flows_30,
    get_normalizing_flows_32,
    get_normalizing_flows_34,
    get_normalizing_flows_36,
    get_normalizing_flows_38,
    get_normalizing_flows_40,
    get_normalizing_flows_42,
    get_normalizing_flows_44,
)
from src.utils.parallel import algorithm_based_run_with_fixed_resources


def generate_ablations(run_no: int, n_particles: int) -> list[dict]:
    # Map flow counts to their corresponding functions
    flow_functions = {
        2: get_normalizing_flows_2,
        4: get_normalizing_flows_4,
        6: get_normalizing_flows_6,
        8: get_normalizing_flows_8,
        10: get_normalizing_flows_10,
        12: get_normalizing_flows_12,
        14: get_normalizing_flows_14,
        16: get_normalizing_flows_16,
        18: get_normalizing_flows_18,
        20: get_normalizing_flows_20,
        22: get_normalizing_flows_22,
        24: get_normalizing_flows_24,
        26: get_normalizing_flows_26,
        28: get_normalizing_flows_28,
        30: get_normalizing_flows_30,
        32: get_normalizing_flows_32,
        34: get_normalizing_flows_34,
        36: get_normalizing_flows_36,
        38: get_normalizing_flows_38,
        40: get_normalizing_flows_40,
        42: get_normalizing_flows_42,
        44: get_normalizing_flows_44,
    }

    # Generate ablations dynamically
    ablations = []
    for n_flows, flow_func in sorted(flow_functions.items()):
        ablations.append(
            {
                "algorithm_suffix": f"N{n_particles}_run{run_no}_nFL{n_flows}_",
                "normalizing_flows": flow_func,
            }
        )

    return ablations


if __name__ == "__main__":
    folder = Path("data") / "raw"
    folder.mkdir(parents=True, exist_ok=True)

    PROBLEMS = ["VSC"]
    ALGORITHMS = [f"RobustBlockVSModel{algo}" for algo in ["VINF"]]

    args = setup_argparse()
    for run_no in range(args.start, args.end):
        for n_particles in NPARTICLES:
            for prob in PROBLEMS:
                train_theta, test_theta = get_train_theta_test_theta(
                    folder, run_no, n_particles, prob, K_LIST, BASE_ALGORITHM
                )

                ablations = generate_ablations(run_no, n_particles)

                algorithm_based_run_with_fixed_resources(
                    run_rjbridge_algorithm_based_ablation,
                    algorithm_size=len(ablations),
                    cores_per_task=1,
                    max_cpu_ratio=0.9,
                    # for run
                    problems=[prob],
                    algorithms=ALGORITHMS * len(ablations),
                    start=args.start,
                    end=args.end,
                    ablations=ablations,
                    # rjbridge
                    train_theta=train_theta,
                    test_theta=test_theta,
                    block_size=BLOCK_SIZE,
                )
