import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.change_point.generate_change_point_smc_samples import (
    generate_fixed_k_smc_samples,
    smc_metadata_path,
    smc_theta_path,
)
from scripts.change_point.change_point_vinfs import build_normalizing_flows
from src.algorithms import ChangePointModelCNF, ChangePointModelVINF
from src.problems import ChangePoint
from src.samplers import RJMCMC


def parse_args():
    parser = argparse.ArgumentParser(description="Run the change-point VINF RJMCMC example.")
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["vinf", "cnf"],
        default="vinf",
        help="Proposal-training strategy: variationally trained flow (vinf) or sample-trained conditional flow (cnf).",
    )
    parser.add_argument("--samples", type=int, default=40000, help="Number of RJMCMC iterations.")
    parser.add_argument("--k-max", type=int, default=10, help="Maximum number of change-points to consider.")
    parser.add_argument("--seed", type=int, default=2222, help="Random seed.")
    parser.add_argument("--flow-iters", type=int, default=10000, help="Training iterations per model-specific flow.")
    parser.add_argument("--flow-samples", type=int, default=256, help="Monte Carlo samples per flow update.")
    parser.add_argument(
        "--flow-device",
        type=str,
        default="cpu",
        help="Device for flow training only, e.g. 'cpu', 'cuda', or 'auto'. Sampling remains on CPU.",
    )
    parser.add_argument("--save-flows-dir", type=str, default="data/flows/change_point", help="Flow checkpoint folder.")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Folder to save chain outputs.")
    parser.add_argument(
        "--within-model-prob",
        type=float,
        default=0.5,
        help="Probability of a within-model move instead of a dimension-changing jump.",
    )
    parser.add_argument(
        "--within-model-scale",
        type=float,
        default=0.35,
        help="Random-walk scale in the flow base space for within-model proposals.",
    )
    parser.add_argument(
        "--aux-scale",
        type=float,
        default=1.0,
        help="Standard deviation of auxiliary variables used in collapsed TRJ birth/death proposals.",
    )
    parser.add_argument(
        "--independent-flows",
        action="store_true",
        help="Use separate model-specific flows instead of the default conditional shared CTP flow.",
    )
    parser.add_argument(
        "--between-model-move",
        type=str,
        choices=["ctp", "latent", "semantic", "semantic_learned"],
        default="ctp",
        help="Dimension-changing move: conditional shared CTP, independent-flow latent append/drop, semantic split/merge, or semantic split/merge with learned rho proposal.",
    )
    parser.add_argument(
        "--split-beta",
        type=float,
        default=2.0,
        help="Beta(a,a) auxiliary distribution parameter for semantic spacing split proportions.",
    )
    parser.add_argument(
        "--semantic-fit-samples",
        type=int,
        default=2048,
        help="Number of flow samples used to fit the learned rho proposal in semantic_learned mode.",
    )
    parser.add_argument(
        "--semantic-min-sigma",
        type=float,
        default=0.25,
        help="Minimum standard deviation for the learned Gaussian proposal on logit(rho).",
    )
    parser.add_argument(
        "--calibration-source",
        type=str,
        choices=["smc", "chain"],
        default="smc",
        help="Training data source for algorithm=cnf: fixed-k SMC posterior samples or a reference RJMCMC chain.",
    )
    parser.add_argument(
        "--calibration-theta",
        type=str,
        default="",
        help="Reference chain used to build calibration draws when --algorithm=cnf --calibration-source=chain.",
    )
    parser.add_argument(
        "--calibration-burn-in",
        type=int,
        default=10000,
        help="Burn-in applied before extracting calibration draws from a reference chain.",
    )
    parser.add_argument(
        "--calibration-per-model",
        type=int,
        default=4000,
        help="Maximum number of calibration draws retained per model for algorithm=cnf.",
    )
    parser.add_argument(
        "--smc-samples-dir",
        type=str,
        default="data/raw/change_point_smc_posterior_samples",
        help="Directory containing fixed-k SMC posterior samples for algorithm=cnf.",
    )
    parser.add_argument(
        "--smc-particles-per-model",
        type=int,
        default=4000,
        help="Number of SMC posterior particles used per fixed-k model for algorithm=cnf.",
    )
    parser.add_argument(
        "--smc-ess-threshold",
        type=float,
        default=0.5,
        help="ESS threshold used if fixed-k SMC samples need to be generated for algorithm=cnf.",
    )
    parser.add_argument(
        "--smc-seed",
        type=int,
        default=2222,
        help="Seed used for fixed-k SMC sample generation/loading in algorithm=cnf mode.",
    )
    parser.add_argument(
        "--smc-force",
        action="store_true",
        help="Regenerate fixed-k SMC posterior samples even if cached files already exist.",
    )
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_flow_initial_state(model, candidates_per_model=8):
    proposal = model.proposal

    candidate_thetas = []
    candidate_scores = []

    for mk in model.getModelKeys():
        theta = proposal.sample_model_parameters_from_flow(mk, candidates_per_model)

        score = model.compute_prior(theta) + model.compute_llh(theta)
        candidate_thetas.append(theta)
        candidate_scores.append(score)

    theta_all = np.vstack(candidate_thetas)
    score_all = np.concatenate(candidate_scores)
    best_idx = int(np.argmax(score_all))
    return theta_all[best_idx]


def build_calibration_draws_from_chain(theta_path: Path, *, burn_in: int, per_model: int, k_col: int, k_max: int):
    theta = np.load(theta_path)
    if burn_in < 0 or burn_in >= theta.shape[0]:
        raise ValueError(f"burn_in must be in [0, {theta.shape[0] - 1}], got {burn_in}")

    theta = theta[burn_in:]
    if theta.shape[0] == 0:
        raise ValueError("No samples remain after burn-in when building calibration draws.")

    k_trace = theta[:, k_col].astype(int)
    selected = []

    for k in range(k_max + 1):
        idx = np.where(k_trace == k)[0]
        if idx.size == 0:
            continue

        take = min(per_model, idx.size)
        chosen = np.random.choice(idx, size=take, replace=False)
        selected.append(theta[chosen])

    if not selected:
        raise ValueError(f"No model samples were found in calibration chain: {theta_path}")

    calibrate_draws = np.vstack(selected)
    np.random.shuffle(calibrate_draws)
    return calibrate_draws


def ensure_smc_samples_exist(args):
    sample_dir = Path(args.smc_samples_dir)
    expected_paths = {
        k: smc_theta_path(
            sample_dir,
            particles=args.smc_particles_per_model,
            seed=args.smc_seed,
            fixed_k=k,
        )
        for k in range(args.k_max + 1)
    }

    missing_paths = [path for path in expected_paths.values() if not path.exists()]
    if args.smc_force or missing_paths:
        generate_fixed_k_smc_samples(
            k_max=args.k_max,
            particles=args.smc_particles_per_model,
            ess_threshold=args.smc_ess_threshold,
            seed=args.smc_seed,
            output_dir=sample_dir,
            within_model_scale=args.within_model_scale,
            force=args.smc_force,
        )

    return expected_paths


def load_smc_timing_metadata(samples_dir: Path, *, particles_per_model: int, seed: int, k_max: int):
    metadata_file = smc_metadata_path(samples_dir, particles=particles_per_model, seed=seed, k_max=k_max)
    if not metadata_file.exists():
        return None
    return json.loads(metadata_file.read_text(encoding="utf-8"))


def build_calibration_draws_from_smc_samples(
    samples_dir: Path,
    *,
    particles_per_model: int,
    seed: int,
    k_max: int,
    per_model: int,
):
    selected = []

    for k in range(k_max + 1):
        theta_path = smc_theta_path(samples_dir, particles=particles_per_model, seed=seed, fixed_k=k)
        if not theta_path.exists():
            raise FileNotFoundError(f"Missing fixed-k SMC sample file: {theta_path}")

        theta_k = np.load(theta_path)
        if theta_k.ndim != 2:
            raise ValueError(f"Expected a 2D theta array in {theta_path}, got shape {theta_k.shape}")
        if theta_k.shape[0] == 0:
            raise ValueError(f"Fixed-k SMC sample file is empty: {theta_path}")

        take = min(per_model, theta_k.shape[0])
        chosen = np.random.choice(theta_k.shape[0], size=take, replace=False)
        selected.append(theta_k[chosen])

    calibrate_draws = np.vstack(selected)
    np.random.shuffle(calibrate_draws)
    return calibrate_draws


def compute_chain_summary(problem, model, theta, ar, *, burn_in=0):
    if burn_in < 0 or burn_in >= theta.shape[0]:
        raise ValueError(f"burn_in must be in [0, {theta.shape[0] - 1}], got {burn_in}")

    theta_eval = theta[burn_in:]
    ar_eval = ar[burn_in:]

    k_col = model.generateRVIndices()["k"][0]
    k_trace = theta_eval[:, k_col].astype(int)
    posterior = np.bincount(k_trace, minlength=problem.k_max + 1).astype(np.float64)
    posterior = posterior / posterior.sum()

    mode_k = int(np.argmax(posterior))
    mode_idx = k_trace == mode_k
    move_rate = 0.0
    model_switch_rate = 0.0
    if theta_eval.shape[0] > 1:
        moved = ~np.isclose(theta_eval[1:], theta_eval[:-1], rtol=1e-10, atol=1e-12).all(axis=1)
        move_rate = float(moved.mean())
        model_switch_rate = float((k_trace[1:] != k_trace[:-1]).mean())

    detail = {
        "burn_in": int(burn_in),
        "mode_k": mode_k,
        "mode_k_probability": float(posterior[mode_k]),
        "mean_acceptance_probability": float(np.mean(np.clip(ar_eval[1:], 0.0, 1.0))) if ar_eval.shape[0] > 1 else 0.0,
        "move_rate": move_rate,
        "model_switch_rate": model_switch_rate,
        "posterior": posterior,
    }

    if mode_idx.any():
        target = problem.target(mode_k)
        active_theta = model._concat_active_parameters(theta_eval[mode_idx], (mode_k,))
        active_theta_t = torch.tensor(active_theta, dtype=torch.float64)
        _, _, change_points = target._decode_and_log_prior(active_theta_t)
        rates = target.posterior_rate_mean(active_theta_t)

        change_points_np = np.asarray(change_points.detach().cpu().tolist(), dtype=np.float64)
        rates_np = np.asarray(rates.detach().cpu().tolist(), dtype=np.float64)

        if mode_k > 0:
            detail["mean_change_points"] = change_points_np.mean(axis=0)
        detail["posterior_mean_rates"] = rates_np.mean(axis=0)

    return detail


def write_chain_outputs(output_dir, stem, theta, prop_theta, llh, log_prior, ar, summary, timing_metadata=None):
    np.save(output_dir / f"{stem}_theta.npy", theta)
    np.save(output_dir / f"{stem}_ptheta.npy", prop_theta)
    np.save(output_dir / f"{stem}_llh.npy", llh)
    np.save(output_dir / f"{stem}_logprior.npy", log_prior)
    np.save(output_dir / f"{stem}_ar.npy", ar)

    summary_lines = ["k,posterior_probability"]
    for k, prob in enumerate(summary["posterior"]):
        summary_lines.append(f"{k},{prob:.6f}")

    summary_path = output_dir / f"{stem}_posterior_k.csv"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    detail_lines = [
        f"burn_in: {summary['burn_in']}",
        f"mode_k: {summary['mode_k']}",
        f"mode_k_probability: {summary['mode_k_probability']:.6f}",
        f"mean_acceptance_probability: {summary['mean_acceptance_probability']:.6f}",
        f"move_rate: {summary['move_rate']:.6f}",
        f"model_switch_rate: {summary['model_switch_rate']:.6f}",
    ]
    if timing_metadata is not None:
        for key, value in timing_metadata.items():
            if isinstance(value, float):
                detail_lines.append(f"{key}: {value:.6f}")
            else:
                detail_lines.append(f"{key}: {value}")
    if "mean_change_points" in summary:
        detail_lines.append(
            "mean_change_points: " + ", ".join(f"{value:.2f}" for value in summary["mean_change_points"])
        )
    if "posterior_mean_rates" in summary:
        detail_lines.append(
            "posterior_mean_rates: " + ", ".join(f"{value:.6f}" for value in summary["posterior_mean_rates"])
        )

    detail_path = output_dir / f"{stem}_summary.txt"
    detail_path.write_text("\n".join(detail_lines), encoding="utf-8")
    timing_path = None
    if timing_metadata is not None:
        timing_path = output_dir / f"{stem}_timing.json"
        timing_path.write_text(json.dumps(timing_metadata, indent=2), encoding="utf-8")
    return summary_path, detail_path, timing_path


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    problem = ChangePoint(k_max=args.k_max)
    use_shared_flow = (args.between_model_move in {"ctp", "semantic", "semantic_learned"}) and not args.independent_flows
    common_model_kwargs = {
        "problem": problem,
        "save_flows_dir": args.save_flows_dir,
        "within_model_prob": args.within_model_prob,
        "within_model_scale": args.within_model_scale,
        "aux_scale": args.aux_scale,
        "use_conditional_shared_flow": use_shared_flow,
        "between_model_move": args.between_model_move,
        "split_beta": args.split_beta,
        "semantic_fit_samples": args.semantic_fit_samples,
        "semantic_min_sigma": args.semantic_min_sigma,
    }
    smc_timing_metadata = None

    setup_started_at = time.perf_counter()
    if args.algorithm == "vinf":
        normalizing_flows = build_normalizing_flows(
            max_iter=args.flow_iters,
            num_samples=args.flow_samples,
            device=args.flow_device,
        )
        model = ChangePointModelVINF(
            normalizing_flows=normalizing_flows,
            **common_model_kwargs,
        )
        sampler = RJMCMC(model)
    else:
        model = ChangePointModelCNF(**common_model_kwargs)
        k_col = model.generateRVIndices()["k"][0]
        if args.calibration_source == "smc":
            ensure_smc_samples_exist(args)
            smc_timing_metadata = load_smc_timing_metadata(
                Path(args.smc_samples_dir),
                particles_per_model=args.smc_particles_per_model,
                seed=args.smc_seed,
                k_max=problem.k_max,
            )
            calibrate_draws = build_calibration_draws_from_smc_samples(
                Path(args.smc_samples_dir),
                particles_per_model=args.smc_particles_per_model,
                seed=args.smc_seed,
                k_max=problem.k_max,
                per_model=args.calibration_per_model,
            )
        else:
            if not args.calibration_theta:
                raise ValueError("--calibration-theta is required when --calibration-source=chain.")
            calibrate_draws = build_calibration_draws_from_chain(
                Path(args.calibration_theta),
                burn_in=args.calibration_burn_in,
                per_model=args.calibration_per_model,
                k_col=k_col,
                k_max=problem.k_max,
            )
        if calibrate_draws.shape[1] != model.dim():
            raise ValueError(
                "Calibration draws have incompatible dimension "
                f"{calibrate_draws.shape[1]} for current model dimension {model.dim()}. "
                "Please provide a reference chain generated by the current change-point implementation."
            )
        sampler = RJMCMC(model, calibrate_draws=calibrate_draws)
    setup_seconds = time.perf_counter() - setup_started_at

    start_theta = select_flow_initial_state(model)
    sampling_started_at = time.perf_counter()
    theta, prop_theta, llh, log_prior, ar = sampler.run(args.samples, start_theta=start_theta)
    sampling_seconds = time.perf_counter() - sampling_started_at

    stem = f"ChangePoint_{type(model).__name__}_NS{args.samples}_seed{args.seed}_K{args.k_max}"
    summary = compute_chain_summary(problem, model, theta, ar, burn_in=0)
    proposal = model.proposal
    timing_metadata = {
        "algorithm": args.algorithm,
        "calibration_source": args.calibration_source if args.algorithm == "cnf" else "vinf_target",
        "setup_seconds": float(setup_seconds),
        "within_model_calibration_seconds": float(getattr(proposal, "within_model_calibration_seconds", 0.0)),
        "td_calibration_seconds": float(getattr(proposal, "td_calibration_seconds", 0.0)),
        "rjmcmc_sampling_seconds": float(sampling_seconds),
        "run_total_seconds": float(setup_seconds + sampling_seconds),
    }
    if smc_timing_metadata is not None:
        timing_metadata["smc_total_recorded_generation_seconds"] = float(
            smc_timing_metadata.get("total_recorded_generation_seconds", 0.0)
        )
        timing_metadata["smc_total_elapsed_seconds_current_run"] = float(
            smc_timing_metadata.get("total_elapsed_seconds_current_run", 0.0)
        )
        timing_metadata["run_total_plus_smc_seconds"] = float(
            timing_metadata["run_total_seconds"] + timing_metadata["smc_total_recorded_generation_seconds"]
        )
    summary_path, detail_path, timing_path = write_chain_outputs(
        output_dir,
        stem,
        theta,
        prop_theta,
        llh,
        log_prior,
        ar,
        summary,
        timing_metadata=timing_metadata,
    )

    print("Posterior model probabilities:")
    for k, prob in enumerate(summary["posterior"]):
        if prob > 0:
            print(f"  k={k:2d}: {prob:.4f}")
    print(f"\nModal k: {summary['mode_k']}")
    print(f"  mode_k_probability: {summary['mode_k_probability']:.6f}")
    print(f"  mean_acceptance_probability: {summary['mean_acceptance_probability']:.6f}")
    print(f"  move_rate: {summary['move_rate']:.6f}")
    print(f"  model_switch_rate: {summary['model_switch_rate']:.6f}")
    if "mean_change_points" in summary:
        print("  mean_change_points: " + ", ".join(f"{value:.2f}" for value in summary["mean_change_points"]))
    if "posterior_mean_rates" in summary:
        print("  posterior_mean_rates: " + ", ".join(f"{value:.6f}" for value in summary["posterior_mean_rates"]))
    print(f"\nSaved chain to: {output_dir}")
    print(f"Saved posterior summary to: {summary_path}")
    print(f"Saved detailed summary to: {detail_path}")
    if timing_path is not None:
        print(f"Saved timing summary to: {timing_path}")


if __name__ == "__main__":
    main()
