from pathlib import Path

import numpy as np

from scripts.fa.fa_base import N_SAMPLES, NPARTICLES, get_train_theta_start_theta, setup_argparse
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
                train_theta, start_theta = get_train_theta_start_theta(folder, run_no, prob, n_particles)

                e = Experiments(args.start, args.end, problems=[prob], algorithms=ALGORITHMS)
                e.run(
                    algorithm_suffix=f"N{n_particles}_pyMC_run{run_no}_",
                    # rjmcmc
                    calibrate_draws=train_theta,
                    start_theta=start_theta,
                    n_samples=N_SAMPLES,
                    # algorithm
                    y_data=Y,
                    normalizing_flows=get_normalizing_flows,
                    # save_flows_dir=str(Path("data") / "flows" / f"{prob}" / f"N{n_particles}"),
                )
