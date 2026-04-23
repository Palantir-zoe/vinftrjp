import argparse
from pathlib import Path

import numpy as np

from scripts.fa.fa_base import N_SAMPLES, NPARTICLES, get_train_theta_start_theta
from scripts.fa.fa_vinfs import get_normalizing_flows
from src.main import Experiments


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run FA RJMCMC with oracle model probabilities and the rejection-free VINF proposal."
    )
    parser.add_argument("--start", type=int, default=1, help="Start run index.")
    parser.add_argument("--end", type=int, default=3, help="End run index.")
    parser.add_argument("--k1-prob", type=float, default=0.88, help="Oracle posterior probability for model k=1.")
    parser.add_argument("--k2-prob", type=float, default=0.12, help="Oracle posterior probability for model k=2.")
    parser.add_argument(
        "--save-flows-dir",
        type=str,
        default="",
        help="Optional directory for loading/saving pretrained FA flows.",
    )
    parser.add_argument(
        "--algorithm-suffix",
        type=str,
        default="RF_",
        help="Suffix inserted into output filenames for this oracle rejection-free run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.start <= 0 or args.end < args.start:
        raise ValueError(f"Invalid run range: start={args.start}, end={args.end}")

    posterior_model_probabilities = {
        (1,): float(args.k1_prob),
        (2,): float(args.k2_prob),
    }
    total_prob = sum(posterior_model_probabilities.values())
    if total_prob <= 0:
        raise ValueError(f"Model probabilities must sum to a positive value, got {posterior_model_probabilities}")
    posterior_model_probabilities = {mk: prob / total_prob for mk, prob in posterior_model_probabilities.items()}

    y_data = np.load(str(Path("data") / "core" / "FA_data.npy"))
    folder = Path("data") / "raw"
    folder.mkdir(parents=True, exist_ok=True)

    problems = ["FA"]
    algorithms = ["FactorAnalysisModelVINFRF"]

    for run_no in range(args.start, args.end + 1):
        for n_particles in NPARTICLES:
            for prob in problems:
                train_theta, start_theta = get_train_theta_start_theta(folder, run_no, prob, n_particles)

                experiments = Experiments(args.start, args.end, problems=[prob], algorithms=algorithms)
                experiments.run(
                    algorithm_suffix=f"N{n_particles}_pyMC_run{run_no}_{args.algorithm_suffix}",
                    calibrate_draws=train_theta,
                    start_theta=start_theta,
                    n_samples=N_SAMPLES,
                    y_data=y_data,
                    normalizing_flows=get_normalizing_flows,
                    posterior_model_probabilities=posterior_model_probabilities,
                    save_flows_dir=args.save_flows_dir,
                )


if __name__ == "__main__":
    main()
