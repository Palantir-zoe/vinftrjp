import argparse
import os
import subprocess
import sys


def setup_argparse():
    parser = argparse.ArgumentParser(description="Run the SAS comparison workflow and save runtime summaries.")
    parser.add_argument("--start", type=int, nargs="?", default=1, help="Start value (default: 1)")
    parser.add_argument("--end", type=int, nargs="?", default=10, help="End value (default: 10)")
    parser.add_argument("--n-samples", type=int, default=10000, help="Number of RJMCMC samples per run.")
    parser.add_argument(
        "--algorithm-suffix",
        type=str,
        default="gpuanneal_",
        help="Suffix appended to SAS output filenames.",
    )
    parser.add_argument(
        "--flow-device",
        type=str,
        default="cuda",
        help="Device for VINF flow training only, e.g. 'cpu', 'cuda', or 'auto'.",
    )
    parser.add_argument(
        "--flow-num-samples",
        type=int,
        default=64,
        help="Number of Monte Carlo samples per flow-training iteration.",
    )
    parser.add_argument(
        "--flow-hidden-layer-size",
        type=int,
        default=128,
        help="Hidden width used by SAS flow training networks.",
    )
    parser.add_argument(
        "--flow-annealing",
        type=str,
        default="true",
        help="Whether to enable beta annealing during SAS flow training (true/false).",
    )
    parser.add_argument(
        "--save-flows-dir",
        type=str,
        default="data/flows/sas_gpuanneal",
        help="Directory used to cache trained SAS flows.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="data/raw/sas_comparison_gpuanneal_summary.csv",
        help="Path to the per-run SAS comparison summary CSV.",
    )
    parser.add_argument(
        "--runtime-summary-csv",
        type=str,
        default="data/raw/sas_comparison_gpuanneal_runtime_summary.csv",
        help="Path to the per-algorithm SAS runtime summary CSV.",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="docs/figures/sas_gpuanneal",
        help="Directory used to save SAS comparison figures.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="true",
        help="Skip runs whose result files already exist (true/false).",
    )
    parser.add_argument(
        "--proposal-run-no",
        type=int,
        default=1,
        help="Experiment index used when plotting SAS proposal figures.",
    )
    parser.add_argument(
        "--proposal-points",
        type=int,
        default=100000,
        help="Calibration sample count used when rendering the SAS proposal figure.",
    )
    return parser.parse_args()


def main():
    try:
        result = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            print("Error: Not in a git repository or git not found")
            sys.exit(1)

        project_dir = result.stdout.strip()
        os.chdir(project_dir)

        args = setup_argparse()

        compare_cmd = [
            sys.executable,
            "scripts/sas/run_sas_comparison_nightly.py",
            f"--start={args.start}",
            f"--end={args.end}",
            f"--n-samples={args.n_samples}",
            f"--algorithm-suffix={args.algorithm_suffix}",
            f"--flow-device={args.flow_device}",
            f"--flow-num-samples={args.flow_num_samples}",
            f"--flow-hidden-layer-size={args.flow_hidden_layer_size}",
            f"--flow-annealing={args.flow_annealing}",
            f"--save-flows-dir={args.save_flows_dir}",
            f"--summary-csv={args.summary_csv}",
            f"--runtime-summary-csv={args.runtime_summary_csv}",
            f"--resume={args.resume}",
        ]

        plot_cmd = [
            sys.executable,
            "scripts/sas/plot_sas_comparison_nightly.py",
            f"--start={args.start}",
            f"--end={args.end}",
            f"--n-samples={args.n_samples}",
            f"--algorithm-suffix={args.algorithm_suffix}",
            f"--figures-dir={args.figures_dir}",
            f"--proposal-run-no={args.proposal_run_no}",
            f"--flow-device={args.flow_device}",
            f"--flow-num-samples={args.flow_num_samples}",
            f"--flow-hidden-layer-size={args.flow_hidden_layer_size}",
            f"--flow-annealing={args.flow_annealing}",
            f"--save-flows-dir={args.save_flows_dir}",
            f"--proposal-points={args.proposal_points}",
        ]

        tasks = [
            ("SAS comparison", compare_cmd),
            ("SAS plotting", plot_cmd),
        ]

        for task_name, cmd in tasks:
            print(f"Executing: {task_name}")
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                print(f"Failed to execute: {task_name}")
                sys.exit(1)

        print(f"Comparison summary: {args.summary_csv}")
        print(f"Runtime summary: {args.runtime_summary_csv}")
        print(f"Figures directory: {args.figures_dir}")
        print("All tasks completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
