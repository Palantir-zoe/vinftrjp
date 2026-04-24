from pathlib import Path

import argparse
import numpy as np

from scripts.fa.fa_base import N_SAMPLES, configure_flow_training, get_train_theta_start_theta
from scripts.fa.fa_vinfs import get_normalizing_flows
from src.algorithms import FactorAnalysisModelVINFIS
from src.problems import FA
from src.samplers import RJMCMC


def get_vinfis_proposal(proposal):
    if hasattr(proposal, "posterior_model_probabilities"):
        return proposal

    for subproposal in getattr(proposal, "ps", []):
        if hasattr(subproposal, "posterior_model_probabilities"):
            return subproposal

    return proposal


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the FA example using flow-based importance sampling to estimate model probabilities."
    )
    parser.add_argument("--run-no", type=int, default=1, help="Run index used to load calibration samples.")
    parser.add_argument(
        "--n-particles",
        type=int,
        default=4000,
        help="Calibration sample size used for the current test run. Default is fixed to 4000.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=N_SAMPLES,
        help="Number of RJMCMC iterations.",
    )
    parser.add_argument(
        "--importance-samples",
        type=int,
        default=4000,
        help="Number of flow samples per candidate model used in the importance-sampling estimator.",
    )
    parser.add_argument(
        "--save-flows-dir",
        type=str,
        default="",
        help="Optional directory for loading/saving pretrained FA flows.",
    )
    parser.add_argument(
        "--algorithm-suffix",
        type=str,
        default="IS_",
        help="Suffix inserted into output filenames for this importance-sampling run.",
    )
    parser.add_argument(
        "--flow-device",
        type=str,
        default="cpu",
        help="Device for flow training only, e.g. 'cpu', 'cuda', or 'auto'.",
    )
    parser.add_argument(
        "--flow-num-samples",
        type=int,
        default=None,
        help="Override the number of Monte Carlo samples per flow-training iteration.",
    )
    parser.add_argument(
        "--flow-hidden-layer-size",
        type=int,
        default=None,
        help="Override the hidden width used by FA flow training networks.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    configure_flow_training(args.flow_device, args.flow_num_samples, args.flow_hidden_layer_size)

    y_data = np.load(str(Path("data") / "core" / "FA_data.npy"))
    folder = Path("data") / "raw"
    folder.mkdir(parents=True, exist_ok=True)

    prob = "FA"
    train_theta, start_theta = get_train_theta_start_theta(folder, args.run_no, prob, args.n_particles)

    problem = FA()
    model = FactorAnalysisModelVINFIS(
        problem=problem,
        y_data=y_data,
        normalizing_flows=get_normalizing_flows,
        importance_num_samples=args.importance_samples,
        save_flows_dir=args.save_flows_dir,
    )

    sampler = RJMCMC(model, calibrate_draws=train_theta)
    theta, prop_theta, llh, log_prior, ar = sampler.run(args.n_samples, start_theta=start_theta)

    stem = (
        f"FA_FactorAnalysisModelVINFIS_N{args.n_particles}_pyMC_run{args.run_no}_"
        f"{args.algorithm_suffix}NS{args.n_samples}"
    )
    np.save(folder / f"{stem}_theta_Exp1.npy", theta)
    np.save(folder / f"{stem}_ptheta_Exp1.npy", prop_theta)
    np.save(folder / f"{stem}_ar_Exp1.npy", ar)
    np.save(folder / f"{stem}_llh_Exp1.npy", llh)
    np.save(folder / f"{stem}_logprior_Exp1.npy", log_prior)

    proposal = get_vinfis_proposal(model.proposal)
    estimated_probs = getattr(proposal, "posterior_model_probabilities", {})
    estimated_log_marginals = getattr(proposal, "estimated_log_marginal_likelihoods", {})
    truth = {(1,): 0.88, (2,): 0.12}

    summary_lines = [
        f"n_particles: {args.n_particles}",
        f"run_no: {args.run_no}",
        f"importance_samples: {args.importance_samples}",
        f"truth_k1: {truth[(1,)]:.6f}",
        f"truth_k2: {truth[(2,)]:.6f}",
    ]
    for mk in sorted(estimated_probs):
        summary_lines.append(f"estimated_prob_{mk}: {estimated_probs[mk]:.6f}")
    for mk in sorted(estimated_log_marginals):
        summary_lines.append(f"log_marginal_{mk}: {estimated_log_marginals[mk]:.6f}")
    if estimated_probs:
        summary_lines.append(f"abs_error_k1: {abs(estimated_probs.get((1,), 0.0) - truth[(1,)]):.6f}")
        summary_lines.append(f"abs_error_k2: {abs(estimated_probs.get((2,), 0.0) - truth[(2,)]):.6f}")

    summary_path = folder / f"{stem}_summary_Exp1.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("Estimated posterior model probabilities from flow importance sampling:")
    for mk in sorted(estimated_probs):
        print(f"  {mk}: {estimated_probs[mk]:.6f}")
    print("Truth:")
    for mk in sorted(truth):
        print(f"  {mk}: {truth[mk]:.6f}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
