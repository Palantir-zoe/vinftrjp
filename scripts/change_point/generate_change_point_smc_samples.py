import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.algorithms import ChangePointModelSMC
from src.problems import ChangePoint
from src.samplers import SMC1


def parse_args():
    parser = argparse.ArgumentParser(description="Generate fixed-k change-point posterior samples using SMC.")
    parser.add_argument("--k-max", type=int, default=10, help="Maximum number of change-points to consider.")
    parser.add_argument("--particles", type=int, default=4000, help="Number of SMC particles per fixed-k model.")
    parser.add_argument("--ess-threshold", type=float, default=0.5, help="ESS threshold used by SMC.")
    parser.add_argument("--seed", type=int, default=2222, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/change_point_smc_posterior_samples",
        help="Directory used to save per-k posterior samples.",
    )
    parser.add_argument(
        "--within-model-scale",
        type=float,
        default=0.35,
        help="Random-walk scale for the fixed-k within-model proposal.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate samples even if output files already exist.",
    )
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def smc_sample_stem(*, particles, seed, fixed_k):
    return f"ChangePoint_ChangePointModelSMC_N{particles}_seed{seed}_K{fixed_k}"


def smc_theta_path(output_dir, *, particles, seed, fixed_k):
    output_dir = Path(output_dir)
    return output_dir / f"{smc_sample_stem(particles=particles, seed=seed, fixed_k=fixed_k)}_theta.npy"


def smc_metadata_path(output_dir, *, particles, seed, k_max):
    output_dir = Path(output_dir)
    return output_dir / f"ChangePoint_ChangePointModelSMC_N{particles}_seed{seed}_KMAX{k_max}_timing.json"


def generate_fixed_k_smc_samples(
    *,
    k_max,
    particles,
    ess_threshold,
    seed,
    output_dir,
    within_model_scale=0.35,
    force=False,
):
    set_seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    problem = ChangePoint(k_max=k_max)
    saved_paths = {}
    per_k_seconds = {}
    reused_models = []
    generated_models = []
    run_started_at = time.perf_counter()

    for fixed_k in range(k_max + 1):
        stem = smc_sample_stem(particles=particles, seed=seed, fixed_k=fixed_k)
        theta_path = output_dir / f"{stem}_theta.npy"
        llh_path = output_dir / f"{stem}_llh.npy"
        logprior_path = output_dir / f"{stem}_logprior.npy"
        summary_path = output_dir / f"{stem}_summary.txt"
        meta_path = output_dir / f"{stem}_timing.json"

        if (not force) and theta_path.exists() and llh_path.exists() and logprior_path.exists():
            print(f"Skipping fixed k={fixed_k}: found existing posterior samples at {theta_path}")
            saved_paths[fixed_k] = theta_path
            reused_models.append(fixed_k)
            if meta_path.exists():
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                per_k_seconds[str(fixed_k)] = metadata.get("elapsed_seconds")
            else:
                per_k_seconds[str(fixed_k)] = None
            continue

        fixed_k_started_at = time.perf_counter()
        model = ChangePointModelSMC(
            problem=problem,
            fixed_k=fixed_k,
            within_model_scale=within_model_scale,
        )
        smc = SMC1(model)
        theta, _, llh, log_prior, _, _ = smc.run(particles, ess_threshold)

        np.save(theta_path, theta)
        np.save(llh_path, llh)
        np.save(logprior_path, log_prior)
        elapsed_seconds = time.perf_counter() - fixed_k_started_at

        summary_lines = [
            f"fixed_k: {fixed_k}",
            f"n_particles: {particles}",
            f"mean_log_likelihood: {float(np.mean(llh)):.6f}",
            f"mean_log_prior: {float(np.mean(log_prior)):.6f}",
            f"elapsed_seconds: {elapsed_seconds:.6f}",
        ]
        summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
        meta_path.write_text(
            json.dumps(
                {
                    "fixed_k": fixed_k,
                    "n_particles": particles,
                    "seed": seed,
                    "ess_threshold": ess_threshold,
                    "within_model_scale": within_model_scale,
                    "elapsed_seconds": elapsed_seconds,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        saved_paths[fixed_k] = theta_path
        generated_models.append(fixed_k)
        per_k_seconds[str(fixed_k)] = elapsed_seconds
        print(f"Saved fixed-k posterior samples for k={fixed_k} to {theta_path}")

    aggregate_metadata = {
        "k_max": k_max,
        "n_particles": particles,
        "seed": seed,
        "ess_threshold": ess_threshold,
        "within_model_scale": within_model_scale,
        "generated_models": generated_models,
        "reused_models": reused_models,
        "per_k_elapsed_seconds": per_k_seconds,
        "total_elapsed_seconds_current_run": time.perf_counter() - run_started_at,
        "total_recorded_generation_seconds": float(
            sum(value for value in per_k_seconds.values() if value is not None)
        ),
    }
    smc_metadata_path(output_dir, particles=particles, seed=seed, k_max=k_max).write_text(
        json.dumps(aggregate_metadata, indent=2),
        encoding="utf-8",
    )

    return saved_paths


def main():
    args = parse_args()
    generate_fixed_k_smc_samples(
        k_max=args.k_max,
        particles=args.particles,
        ess_threshold=args.ess_threshold,
        seed=args.seed,
        output_dir=args.output_dir,
        within_model_scale=args.within_model_scale,
        force=args.force,
    )


if __name__ == "__main__":
    main()
