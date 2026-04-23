import argparse
from pathlib import Path

import numpy as np
import torch

from scripts.change_point.change_point_vinfs import build_normalizing_flows
from src.algorithms import ChangePointModelVINF
from src.problems import ChangePoint
from src.samplers import RJMCMC


def parse_args():
    parser = argparse.ArgumentParser(description="Run the change-point VINF RJMCMC example.")
    parser.add_argument("--samples", type=int, default=40000, help="Number of RJMCMC iterations.")
    parser.add_argument("--k-max", type=int, default=10, help="Maximum number of change-points to consider.")
    parser.add_argument("--seed", type=int, default=2222, help="Random seed.")
    parser.add_argument("--flow-iters", type=int, default=10000, help="Training iterations per model-specific flow.")
    parser.add_argument("--flow-samples", type=int, default=256, help="Monte Carlo samples per flow update.")
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


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalizing_flows = build_normalizing_flows(
        max_iter=args.flow_iters,
        num_samples=args.flow_samples,
    )

    problem = ChangePoint(k_max=args.k_max)
    model = ChangePointModelVINF(
        problem=problem,
        normalizing_flows=normalizing_flows,
        save_flows_dir=args.save_flows_dir,
        within_model_prob=args.within_model_prob,
        within_model_scale=args.within_model_scale,
        aux_scale=args.aux_scale,
        use_conditional_shared_flow=not args.independent_flows,
    )

    sampler = RJMCMC(model)
    start_theta = select_flow_initial_state(model)
    theta, prop_theta, llh, log_prior, ar = sampler.run(args.samples, start_theta=start_theta)

    stem = f"ChangePoint_ChangePointModelVINF_NS{args.samples}_seed{args.seed}_K{args.k_max}"
    np.save(output_dir / f"{stem}_theta.npy", theta)
    np.save(output_dir / f"{stem}_ptheta.npy", prop_theta)
    np.save(output_dir / f"{stem}_llh.npy", llh)
    np.save(output_dir / f"{stem}_logprior.npy", log_prior)
    np.save(output_dir / f"{stem}_ar.npy", ar)

    k_col = model.generateRVIndices()["k"][0]
    k_trace = theta[:, k_col].astype(int)
    posterior = np.bincount(k_trace, minlength=problem.k_max + 1)
    posterior = posterior / posterior.sum()

    summary_lines = ["k,posterior_probability"]
    for k, prob in enumerate(posterior):
        summary_lines.append(f"{k},{prob:.6f}")

    summary_path = output_dir / f"{stem}_posterior_k.csv"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    mode_k = int(np.argmax(posterior))
    mode_idx = k_trace == mode_k
    detail_lines = [f"mode_k: {mode_k}", f"mode_k_probability: {posterior[mode_k]:.6f}"]

    if mode_idx.any():
        target = problem.target(mode_k)
        active_theta = model._concat_active_parameters(theta[mode_idx], (mode_k,))
        active_theta_t = torch.tensor(active_theta, dtype=torch.float64)
        _, _, change_points = target._decode_and_log_prior(active_theta_t)
        rates = target.posterior_rate_mean(active_theta_t)

        change_points_np = np.asarray(change_points.detach().cpu().tolist(), dtype=np.float64)
        rates_np = np.asarray(rates.detach().cpu().tolist(), dtype=np.float64)

        if mode_k > 0:
            detail_lines.append(
                "mean_change_points: " + ", ".join(f"{value:.2f}" for value in change_points_np.mean(axis=0))
            )
        detail_lines.append("posterior_mean_rates: " + ", ".join(f"{value:.6f}" for value in rates_np.mean(axis=0)))

    detail_path = output_dir / f"{stem}_summary.txt"
    detail_path.write_text("\n".join(detail_lines), encoding="utf-8")

    print("Posterior model probabilities:")
    for k, prob in enumerate(posterior):
        if prob > 0:
            print(f"  k={k:2d}: {prob:.4f}")
    print(f"\nModal k: {mode_k}")
    for line in detail_lines[1:]:
        print(f"  {line}")
    print(f"\nSaved chain to: {output_dir}")
    print(f"Saved posterior summary to: {summary_path}")
    print(f"Saved detailed summary to: {detail_path}")


if __name__ == "__main__":
    main()
