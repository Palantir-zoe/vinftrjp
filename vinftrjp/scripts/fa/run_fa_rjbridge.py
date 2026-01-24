from pathlib import Path

import numpy as np

from scripts.fa.fa_base import BLOCK_SIZE, NPARTICLES, get_train_theta_test_theta, setup_argparse
from scripts.fa.fa_vinfs import get_normalizing_flows
from src.main import Experiments

if __name__ == "__main__":
    Y = np.load(str(Path("data") / "core" / "FA_data.npy"))
    folder = Path("data") / "raw"
    folder.mkdir(parents=True, exist_ok=True)

    PROBLEMS = ["FA"]
    ALGORITHMS = [f"FactorAnalysisModel{algo}" for algo in ["LW", "AF", "NF", "VINF"]]

    args = setup_argparse()
    for run_no in range(args.start, args.end + 1):
        for n_particles in NPARTICLES:
            for prob in PROBLEMS:
                train_theta, test_theta = get_train_theta_test_theta(folder, run_no, prob, n_particles)

                e = Experiments(args.start, args.end, problems=[prob], algorithms=ALGORITHMS)
                e.run(
                    algorithm_suffix=f"N{n_particles}_pyMC_NUTS_run{run_no}_",
                    # rjbridge
                    train_theta=train_theta,
                    test_theta=test_theta,
                    block_size=BLOCK_SIZE,
                    # algorithm
                    y_data=Y,
                    normalizing_flows=get_normalizing_flows,
                    # save_flows_dir=str(Path("data") / "flows" / f"{prob}" / f"N{n_particles}"),
                )
