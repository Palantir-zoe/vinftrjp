import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.change_point.generate_change_point_smc_samples import smc_metadata_path
from scripts.change_point.run_change_point_rjmcmc import compute_chain_summary
from src.algorithms import ChangePointModelCNF
from src.problems import ChangePoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare change-point collapsed shared CTP trained by VINF versus fixed-k SMC-trained CNF."
    )
    parser.add_argument("--samples", type=int, default=100000, help="Number of RJMCMC iterations for both algorithms.")
    parser.add_argument("--burn-in", type=int, default=10000, help="Burn-in used in the comparison summary.")
    parser.add_argument("--k-max", type=int, default=10, help="Maximum number of change-points to consider.")
    parser.add_argument("--seed", type=int, default=2222, help="Random seed for both RJMCMC runs.")
    parser.add_argument("--flow-iters", type=int, default=20000, help="Flow training iterations for VINF.")
    parser.add_argument("--flow-samples", type=int, default=256, help="Monte Carlo samples per VINF flow update.")
    parser.add_argument("--flow-device", type=str, default="cpu", help="Device used for VINF flow training only.")
    parser.add_argument(
        "--save-flows-root",
        type=str,
        default="data/flows/change_point_vinf_vs_cnf",
        help="Root directory used to save trained flow checkpoints.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/raw/change_point_vinf_vs_cnf",
        help="Root directory used to save RJMCMC outputs and comparison summaries.",
    )
    parser.add_argument(
        "--smc-samples-dir",
        type=str,
        default="data/raw/change_point_smc_posterior_samples",
        help="Directory containing fixed-k SMC posterior samples used to train the CNF baseline.",
    )
    parser.add_argument(
        "--smc-particles-per-model",
        type=int,
        default=4000,
        help="Number of SMC posterior samples used per fixed-k model.",
    )
    parser.add_argument("--smc-ess-threshold", type=float, default=0.5, help="ESS threshold used in fixed-k SMC.")
    parser.add_argument("--smc-seed", type=int, default=2222, help="Seed used for fixed-k SMC generation/loading.")
    parser.add_argument("--smc-force", action="store_true", help="Regenerate fixed-k SMC samples even if cached.")
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
        help="Reference scale for inactive coordinates in collapsed shared CTP proposals.",
    )
    parser.add_argument(
        "--calibration-per-model",
        type=int,
        default=4000,
        help="Number of fixed-k SMC posterior samples retained per model for CNF flow training.",
    )
    parser.add_argument("--rerun-vinf", action="store_true", help="Force rerunning the VINF chain.")
    parser.add_argument("--rerun-cnf", action="store_true", help="Force rerunning the CNF chain.")
    return parser.parse_args()


def run_command(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_stem(model_name, samples, seed, k_max):
    return f"ChangePoint_{model_name}_NS{samples}_seed{seed}_K{k_max}"


def expected_theta_path(output_dir, model_name, samples, seed, k_max):
    return Path(output_dir) / f"{build_stem(model_name, samples, seed, k_max)}_theta.npy"


def expected_timing_path(output_dir, model_name, samples, seed, k_max):
    return Path(output_dir) / f"{build_stem(model_name, samples, seed, k_max)}_timing.json"


def maybe_run_vinf(args, vinf_output_dir, vinf_flows_dir):
    theta_path = expected_theta_path(vinf_output_dir, "ChangePointModelVINF", args.samples, args.seed, args.k_max)
    timing_path = expected_timing_path(vinf_output_dir, "ChangePointModelVINF", args.samples, args.seed, args.k_max)
    if theta_path.exists() and timing_path.exists() and not args.rerun_vinf:
        print(f"Reusing existing VINF chain: {theta_path}")
        return

    cmd = [
        sys.executable,
        "-m",
        "scripts.change_point.run_change_point_rjmcmc",
        "--algorithm",
        "vinf",
        "--samples",
        str(args.samples),
        "--k-max",
        str(args.k_max),
        "--seed",
        str(args.seed),
        "--flow-iters",
        str(args.flow_iters),
        "--flow-samples",
        str(args.flow_samples),
        "--flow-device",
        args.flow_device,
        "--save-flows-dir",
        str(vinf_flows_dir),
        "--output-dir",
        str(vinf_output_dir),
        "--within-model-prob",
        str(args.within_model_prob),
        "--within-model-scale",
        str(args.within_model_scale),
        "--aux-scale",
        str(args.aux_scale),
        "--between-model-move",
        "ctp",
    ]
    run_command(cmd)


def maybe_run_cnf(args, cnf_output_dir, cnf_flows_dir):
    theta_path = expected_theta_path(cnf_output_dir, "ChangePointModelCNF", args.samples, args.seed, args.k_max)
    timing_path = expected_timing_path(cnf_output_dir, "ChangePointModelCNF", args.samples, args.seed, args.k_max)
    smc_timing_path = smc_metadata_path(
        args.smc_samples_dir,
        particles=args.smc_particles_per_model,
        seed=args.smc_seed,
        k_max=args.k_max,
    )
    need_smc_timing = True
    if smc_timing_path.exists():
        smc_timing = json.loads(smc_timing_path.read_text(encoding="utf-8"))
        need_smc_timing = float(smc_timing.get("total_recorded_generation_seconds", 0.0)) <= 0.0

    need_cnf_rerun_for_timing = True
    if timing_path.exists():
        cnf_timing = json.loads(timing_path.read_text(encoding="utf-8"))
        need_cnf_rerun_for_timing = "run_total_plus_smc_seconds" not in cnf_timing

    if theta_path.exists() and timing_path.exists() and not args.rerun_cnf and not need_smc_timing and not need_cnf_rerun_for_timing:
        print(f"Reusing existing CNF chain: {theta_path}")
        return

    cmd = [
        sys.executable,
        "-m",
        "scripts.change_point.run_change_point_rjmcmc",
        "--algorithm",
        "cnf",
        "--calibration-source",
        "smc",
        "--samples",
        str(args.samples),
        "--k-max",
        str(args.k_max),
        "--seed",
        str(args.seed),
        "--save-flows-dir",
        str(cnf_flows_dir),
        "--output-dir",
        str(cnf_output_dir),
        "--within-model-prob",
        str(args.within_model_prob),
        "--within-model-scale",
        str(args.within_model_scale),
        "--aux-scale",
        str(args.aux_scale),
        "--between-model-move",
        "ctp",
        "--smc-samples-dir",
        str(args.smc_samples_dir),
        "--smc-particles-per-model",
        str(args.smc_particles_per_model),
        "--smc-ess-threshold",
        str(args.smc_ess_threshold),
        "--smc-seed",
        str(args.smc_seed),
        "--calibration-per-model",
        str(args.calibration_per_model),
    ]
    if args.smc_force or need_smc_timing:
        cmd.append("--smc-force")
    run_command(cmd)


def load_chain_summary(problem, theta_path, ar_path, *, burn_in):
    theta = np.load(theta_path)
    ar = np.load(ar_path)
    model = ChangePointModelCNF(problem=problem)
    return compute_chain_summary(problem, model, theta, ar, burn_in=burn_in)


def load_timing_metadata(timing_path: Path):
    if not timing_path.exists():
        return {}
    return json.loads(timing_path.read_text(encoding="utf-8"))


def write_comparison(output_dir, summaries, timings, *, k_max):
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_lines = [
        ",".join(
            [
                "algorithm",
                "burn_in",
                "mode_k",
                "mode_k_probability",
                "mean_acceptance_probability",
                "move_rate",
                "model_switch_rate",
                "setup_seconds",
                "within_model_calibration_seconds",
                "td_calibration_seconds",
                "rjmcmc_sampling_seconds",
                "run_total_seconds",
                "smc_total_recorded_generation_seconds",
                "run_total_plus_smc_seconds",
                *[f"p_k_{k}" for k in range(k_max + 1)],
            ]
        )
    ]
    txt_lines = []

    for algorithm, summary in summaries.items():
        timing = timings.get(algorithm, {})
        row = [
            algorithm,
            str(summary["burn_in"]),
            str(summary["mode_k"]),
            f"{summary['mode_k_probability']:.6f}",
            f"{summary['mean_acceptance_probability']:.6f}",
            f"{summary['move_rate']:.6f}",
            f"{summary['model_switch_rate']:.6f}",
            f"{float(timing.get('setup_seconds', 0.0)):.6f}",
            f"{float(timing.get('within_model_calibration_seconds', 0.0)):.6f}",
            f"{float(timing.get('td_calibration_seconds', 0.0)):.6f}",
            f"{float(timing.get('rjmcmc_sampling_seconds', 0.0)):.6f}",
            f"{float(timing.get('run_total_seconds', 0.0)):.6f}",
            f"{float(timing.get('smc_total_recorded_generation_seconds', 0.0)):.6f}",
            f"{float(timing.get('run_total_plus_smc_seconds', timing.get('run_total_seconds', 0.0))):.6f}",
            *[f"{prob:.6f}" for prob in summary["posterior"]],
        ]
        csv_lines.append(",".join(row))

        txt_lines.append(f"[{algorithm}]")
        txt_lines.append(f"burn_in: {summary['burn_in']}")
        txt_lines.append(f"mode_k: {summary['mode_k']}")
        txt_lines.append(f"mode_k_probability: {summary['mode_k_probability']:.6f}")
        txt_lines.append(f"mean_acceptance_probability: {summary['mean_acceptance_probability']:.6f}")
        txt_lines.append(f"move_rate: {summary['move_rate']:.6f}")
        txt_lines.append(f"model_switch_rate: {summary['model_switch_rate']:.6f}")
        txt_lines.append(f"setup_seconds: {float(timing.get('setup_seconds', 0.0)):.6f}")
        txt_lines.append(
            f"within_model_calibration_seconds: {float(timing.get('within_model_calibration_seconds', 0.0)):.6f}"
        )
        txt_lines.append(f"td_calibration_seconds: {float(timing.get('td_calibration_seconds', 0.0)):.6f}")
        txt_lines.append(f"rjmcmc_sampling_seconds: {float(timing.get('rjmcmc_sampling_seconds', 0.0)):.6f}")
        txt_lines.append(f"run_total_seconds: {float(timing.get('run_total_seconds', 0.0)):.6f}")
        if "smc_total_recorded_generation_seconds" in timing:
            txt_lines.append(
                "smc_total_recorded_generation_seconds: "
                f"{float(timing.get('smc_total_recorded_generation_seconds', 0.0)):.6f}"
            )
        if "run_total_plus_smc_seconds" in timing:
            txt_lines.append(f"run_total_plus_smc_seconds: {float(timing['run_total_plus_smc_seconds']):.6f}")
        txt_lines.append("posterior_k: " + ", ".join(f"k={k}:{prob:.6f}" for k, prob in enumerate(summary["posterior"])))
        if "mean_change_points" in summary:
            txt_lines.append(
                "mean_change_points: " + ", ".join(f"{value:.2f}" for value in summary["mean_change_points"])
            )
        if "posterior_mean_rates" in summary:
            txt_lines.append(
                "posterior_mean_rates: " + ", ".join(f"{value:.6f}" for value in summary["posterior_mean_rates"])
            )
        txt_lines.append("")

    csv_path = output_dir / "comparison.csv"
    txt_path = output_dir / "comparison.txt"
    csv_path.write_text("\n".join(csv_lines), encoding="utf-8")
    txt_path.write_text("\n".join(txt_lines).strip() + "\n", encoding="utf-8")
    return csv_path, txt_path


def main():
    args = parse_args()

    output_root = Path(args.output_root)
    flow_root = Path(args.save_flows_root)

    vinf_output_dir = output_root / "vinf"
    cnf_output_dir = output_root / "cnf"
    compare_output_dir = output_root / "comparison"

    vinf_flows_dir = flow_root / "vinf"
    cnf_flows_dir = flow_root / "cnf"

    maybe_run_vinf(args, vinf_output_dir, vinf_flows_dir)
    maybe_run_cnf(args, cnf_output_dir, cnf_flows_dir)

    problem = ChangePoint(k_max=args.k_max)
    vinf_stem = build_stem("ChangePointModelVINF", args.samples, args.seed, args.k_max)
    cnf_stem = build_stem("ChangePointModelCNF", args.samples, args.seed, args.k_max)

    summaries = {
        "vinf_ctp": load_chain_summary(
            problem,
            vinf_output_dir / f"{vinf_stem}_theta.npy",
            vinf_output_dir / f"{vinf_stem}_ar.npy",
            burn_in=args.burn_in,
        ),
        "smc_cnf_ctp": load_chain_summary(
            problem,
            cnf_output_dir / f"{cnf_stem}_theta.npy",
            cnf_output_dir / f"{cnf_stem}_ar.npy",
            burn_in=args.burn_in,
        ),
    }
    timings = {
        "vinf_ctp": load_timing_metadata(
            expected_timing_path(vinf_output_dir, "ChangePointModelVINF", args.samples, args.seed, args.k_max)
        ),
        "smc_cnf_ctp": load_timing_metadata(
            expected_timing_path(cnf_output_dir, "ChangePointModelCNF", args.samples, args.seed, args.k_max)
        ),
    }

    csv_path, txt_path = write_comparison(compare_output_dir, summaries, timings, k_max=args.k_max)

    print("Comparison summary:")
    for algorithm, summary in summaries.items():
        timing = timings.get(algorithm, {})
        total_seconds = float(timing.get("run_total_plus_smc_seconds", timing.get("run_total_seconds", 0.0)))
        print(
            f"  {algorithm}: mode_k={summary['mode_k']}, "
            f"p={summary['mode_k_probability']:.4f}, "
            f"acc_prob={summary['mean_acceptance_probability']:.4f}, "
            f"switch_rate={summary['model_switch_rate']:.4f}, "
            f"total_seconds={total_seconds:.2f}"
        )
    print(f"Saved CSV comparison to: {csv_path}")
    print(f"Saved text comparison to: {txt_path}")


if __name__ == "__main__":
    main()
