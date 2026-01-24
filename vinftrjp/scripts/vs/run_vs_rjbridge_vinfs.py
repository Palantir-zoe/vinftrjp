from pathlib import Path

from scripts.vs.vs_base import (
    BASE_ALGORITHM,
    BLOCK_SIZE,
    K_LIST,
    NPARTICLES,
    get_train_theta_test_theta,
    setup_argparse,
)
from scripts.vs.vs_vinfs import get_normalizing_flows
from src.main import Experiments

if __name__ == "__main__":
    folder = Path("data") / "raw"
    folder.mkdir(parents=True, exist_ok=True)

    PROBLEMS = ["VS"]
    ALGORITHMS = [f"RobustBlockVSModel{algo}" for algo in ["IndivAF", "IndivRQ", "CNF", "Naive", "IndivVINF"]]

    args = setup_argparse()
    for run_no in range(args.start, args.end + 1):
        for n_particles in NPARTICLES:
            for prob in PROBLEMS:
                train_theta, test_theta = get_train_theta_test_theta(
                    folder, run_no, n_particles, prob, K_LIST, BASE_ALGORITHM
                )

                e = Experiments(args.start, args.end, problems=[prob], algorithms=ALGORITHMS)
                e.run(
                    algorithm_suffix=f"N{n_particles}_run{run_no}_",
                    # rjbridge
                    train_theta=train_theta,
                    test_theta=test_theta,
                    block_size=BLOCK_SIZE,
                    # algorithm
                    normalizing_flows=get_normalizing_flows,
                    # save_flows_dir=str(Path("data") / "flows" / f"{prob}" / f"N{n_particles}"),
                )
