from pathlib import Path

from scripts.vs.vs_base import K_LIST, NPARTICLES, generate_train_theta_test_theta, setup_argparse

if __name__ == "__main__":
    folder = Path("data") / "raw"
    folder.mkdir(parents=True, exist_ok=True)

    PROBLEMS = ["VS", "VSC"]
    BASE_ALGORITHM = "RobustBlockVSModelIndivSMC"

    args = setup_argparse()
    for run_no in range(args.start, args.end + 1):
        for n_particles in NPARTICLES:
            for prob in PROBLEMS:
                generate_train_theta_test_theta(folder, run_no, n_particles, prob, K_LIST, BASE_ALGORITHM)
