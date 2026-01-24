from pathlib import Path

import numpy as np

from scripts.fa.fa_base import (
    NPARTICLES,
    generate_train_theta_test_theta_k1,
    generate_train_theta_test_theta_k2,
    setup_argparse,
)

if __name__ == "__main__":
    Y = np.load(str(Path("data") / "core" / "FA_data.npy"))
    folder = Path("data") / "raw"
    folder.mkdir(parents=True, exist_ok=True)

    PROBLEMS = ["FA"]
    CORES = 4
    CHAINS = 4

    args = setup_argparse()
    for run_no in range(args.start, args.end + 1):
        for idx, prob in enumerate(PROBLEMS):
            SEED = 123456789 + run_no * len(PROBLEMS) * CHAINS + idx

            for n_particles in NPARTICLES:
                SMC_SAMPLE_KWARGS = {
                    "cores": CORES,
                    "random_seed": [SEED + i for i in range(CHAINS)],
                    "draws": n_particles,
                    "chains": CHAINS,
                    "return_inferencedata": True,
                }
                generate_train_theta_test_theta_k1(folder, run_no, n_particles, prob, Y, SMC_SAMPLE_KWARGS)

                NUTS_SAMPLE_KWARGS = {
                    "cores": CORES,
                    "init": "adapt_diag",
                    "random_seed": [SEED + i for i in range(CHAINS)],
                    "draws": n_particles,
                    "tune": 1000,
                    "chains": CHAINS,
                    "return_inferencedata": True,
                }
                generate_train_theta_test_theta_k2(folder, run_no, n_particles, prob, Y, NUTS_SAMPLE_KWARGS)
