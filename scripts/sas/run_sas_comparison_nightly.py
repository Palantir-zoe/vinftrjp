import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.sas.sas_base import DEFAULT_DATA_DICT, N_SAMPLES, configure_flow_training
from scripts.sas.sas_vinfs import get_normalizing_flows
from src.algorithms import get_algorithm
from src.main import Experiment, Experiments
from src.problems import get_problem
from src.samplers import RJMCMC
from src.vi_nflows import resolve_flow_training_device

K_COLUMN = 1


def _parse_bool(value):
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 10-way SAS algorithm comparisons with GPU-enabled VINF flow training."
    )
    parser.add_argument("--start", type=int, default=1, help="First experiment index to run.")
    parser.add_argument("--end", type=int, default=10, help="Last experiment index to run.")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES, help="Number of RJMCMC samples per run.")
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
        default=None,
        help="Override the number of Monte Carlo samples per flow-training iteration.",
    )
    parser.add_argument(
        "--flow-hidden-layer-size",
        type=int,
        default=None,
        help="Override the hidden width used by SAS flow training networks.",
    )
    parser.add_argument(
        "--flow-annealing",
        type=_parse_bool,
        default=True,
        help="Whether to enable beta annealing during SAS flow training.",
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
        help="Path to the per-run summary CSV written during execution.",
    )
    parser.add_argument(
        "--runtime-summary-csv",
        type=str,
        default="data/raw/sas_comparison_gpuanneal_runtime_summary.csv",
        help="Path to the per-algorithm runtime summary CSV.",
    )
    parser.add_argument(
        "--resume",
        type=_parse_bool,
        default=True,
        help="Skip jobs whose output files already exist.",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class Job:
    algorithm: str
    run_no: int


def raw_path(algorithm, suffix, n_samples, kind, run_no):
    return Path("data") / "raw" / f"SAS_{algorithm}_{suffix}NS{n_samples}_{kind}_Exp{run_no}.npy"


def job_key(job):
    return (job.algorithm, int(job.run_no))


def should_count_posterior_sample_time(algorithm):
    return algorithm in {"ToyModelAF", "ToyModelNF"}


def extract_td_proposal(proposal):
    if hasattr(proposal, "td_calibration_seconds"):
        return proposal
    for subproposal in getattr(proposal, "ps", []):
        if hasattr(subproposal, "td_calibration_seconds"):
            return subproposal
    return None


def accepted_move_rate(theta, prop_theta):
    if theta.shape[0] <= 1:
        return 0.0
    moved = np.any(np.abs(theta[1:] - prop_theta[1:]) > 1e-12, axis=1)
    return float(np.mean(moved))


def effective_sample_size_indicator(samples):
    x = np.asarray(samples, dtype=np.float64)
    if x.size < 4:
        return float(x.size)

    x = x - x.mean()
    variance = np.dot(x, x) / x.size
    if variance <= 0:
        return 0.0

    n = x.size
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2

    freq = np.fft.rfft(x, n=fft_size)
    acov = np.fft.irfft(freq * np.conjugate(freq), n=fft_size)[:n]
    acov = acov / np.arange(n, 0, -1)
    acor = acov / acov[0]

    tau = 1.0
    for lag in range(1, n - 1, 2):
        pair_sum = acor[lag] + acor[lag + 1]
        if pair_sum <= 0:
            break
        tau += 2.0 * pair_sum

    tau = max(tau, 1.0)
    return float(n / tau)


def summarise_run(
    job,
    args,
    runtime_seconds=None,
    resumed=False,
    posterior_sample_generation_seconds=None,
    td_proposal_fit_or_train_seconds=None,
    rjmcmc_sampling_seconds=None,
):
    theta = np.load(raw_path(job.algorithm, args.algorithm_suffix, args.n_samples, "theta", job.run_no))
    prop_theta = np.load(raw_path(job.algorithm, args.algorithm_suffix, args.n_samples, "ptheta", job.run_no))

    model_indicator = theta[:, K_COLUMN]
    return {
        "algorithm": job.algorithm,
        "run_no": job.run_no,
        "n_samples": args.n_samples,
        "algorithm_suffix": args.algorithm_suffix,
        "flow_device": args.flow_device if job.algorithm == "ToyModelVINF" else "",
        "flow_num_samples": args.flow_num_samples if job.algorithm == "ToyModelVINF" else "",
        "flow_hidden_layer_size": args.flow_hidden_layer_size if job.algorithm == "ToyModelVINF" else "",
        "flow_annealing": args.flow_annealing if job.algorithm == "ToyModelVINF" else "",
        "runtime_seconds": "" if runtime_seconds is None else float(runtime_seconds),
        "resumed": bool(resumed),
        "posterior_sample_generation_seconds": (
            "" if posterior_sample_generation_seconds is None else float(posterior_sample_generation_seconds)
        ),
        "td_proposal_fit_or_train_seconds": (
            "" if td_proposal_fit_or_train_seconds is None else float(td_proposal_fit_or_train_seconds)
        ),
        "rjmcmc_sampling_seconds": "" if rjmcmc_sampling_seconds is None else float(rjmcmc_sampling_seconds),
        "model2_probability_mean": float(np.mean(model_indicator)),
        "model2_probability_final_running": float(np.mean(model_indicator)),
        "accepted_move_rate": accepted_move_rate(theta, prop_theta),
        "ess_model2_indicator": effective_sample_size_indicator(model_indicator),
    }


def load_existing_summary(summary_csv):
    if not summary_csv.exists():
        return {}

    with summary_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = {}
        for row in reader:
            rows[(row["algorithm"], int(row["run_no"]))] = row
        return rows


def write_summary(rows, summary_csv):
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "algorithm",
        "run_no",
        "n_samples",
        "algorithm_suffix",
        "flow_device",
        "flow_num_samples",
        "flow_hidden_layer_size",
        "flow_annealing",
        "runtime_seconds",
        "resumed",
        "posterior_sample_generation_seconds",
        "td_proposal_fit_or_train_seconds",
        "rjmcmc_sampling_seconds",
        "model2_probability_mean",
        "model2_probability_final_running",
        "accepted_move_rate",
        "ess_model2_indicator",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_runtime_summary(rows, runtime_summary_csv):
    runtime_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["algorithm"], []).append(row)

    fieldnames = [
        "algorithm",
        "n_runs",
        "n_fresh_runs",
        "mean_runtime_seconds",
        "median_runtime_seconds",
        "max_runtime_seconds",
        "mean_posterior_sample_generation_seconds",
        "mean_td_proposal_fit_or_train_seconds",
        "mean_rjmcmc_sampling_seconds",
        "mean_model2_probability",
        "mean_accepted_move_rate",
        "mean_ess_model2_indicator",
    ]
    with runtime_summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for algorithm, algo_rows in grouped.items():
            runtime_values = []
            fresh_runs = 0
            for row in algo_rows:
                value = row.get("runtime_seconds", "")
                if value not in {"", None}:
                    runtime_values.append(float(value))
                if str(row.get("resumed", "")).lower() not in {"true", "1"}:
                    fresh_runs += 1

            writer.writerow(
                {
                    "algorithm": algorithm,
                    "n_runs": len(algo_rows),
                    "n_fresh_runs": fresh_runs,
                    "mean_runtime_seconds": float(np.mean(runtime_values)) if runtime_values else "",
                    "median_runtime_seconds": float(np.median(runtime_values)) if runtime_values else "",
                    "max_runtime_seconds": float(np.max(runtime_values)) if runtime_values else "",
                    "mean_posterior_sample_generation_seconds": _mean_float_field(
                        algo_rows, "posterior_sample_generation_seconds"
                    ),
                    "mean_td_proposal_fit_or_train_seconds": _mean_float_field(
                        algo_rows, "td_proposal_fit_or_train_seconds"
                    ),
                    "mean_rjmcmc_sampling_seconds": _mean_float_field(algo_rows, "rjmcmc_sampling_seconds"),
                    "mean_model2_probability": float(np.mean([float(row["model2_probability_mean"]) for row in algo_rows])),
                    "mean_accepted_move_rate": float(np.mean([float(row["accepted_move_rate"]) for row in algo_rows])),
                    "mean_ess_model2_indicator": float(np.mean([float(row["ess_model2_indicator"]) for row in algo_rows])),
                }
            )


def _mean_float_field(rows, field_name):
    values = []
    for row in rows:
        value = row.get(field_name, "")
        if value not in {"", None}:
            values.append(float(value))
    return float(np.mean(values)) if values else ""


def run_job(job, args, seed, run_device):
    experiment = Experiment("SAS", index=job.run_no, seed=seed, device=run_device)
    experiment.set_random_seed(seed, deterministic=True)

    problem = get_problem("SAS")
    kwargs = {}
    if job.algorithm == "ToyModelVINF":
        kwargs["normalizing_flows"] = get_normalizing_flows
        kwargs["save_flows_dir"] = args.save_flows_dir

    model = get_algorithm(job.algorithm, problem=problem, **kwargs)

    posterior_sample_generation_seconds = None
    sample_start = time.perf_counter()
    calibrate_draws = model.draw_perfect(args.n_samples)
    sample_elapsed = time.perf_counter() - sample_start
    if should_count_posterior_sample_time(job.algorithm):
        posterior_sample_generation_seconds = sample_elapsed

    fit_start = time.perf_counter()
    rjmcmc = RJMCMC(model, calibrate_draws=calibrate_draws)
    fit_elapsed = time.perf_counter() - fit_start
    td_proposal = extract_td_proposal(model.proposal)
    td_fit_or_train_seconds = getattr(td_proposal, "td_calibration_seconds", fit_elapsed)

    sampling_start = time.perf_counter()
    final_theta, prop_theta, _, _, ar = rjmcmc.run(args.n_samples, None)
    sampling_elapsed = time.perf_counter() - sampling_start

    base_name = "SAS_{}_{}NS{}_{}_Exp{}.npy"
    np.save(Path(experiment._folder_raw) / base_name.format(job.algorithm, args.algorithm_suffix, args.n_samples, "theta", job.run_no), final_theta)
    np.save(Path(experiment._folder_raw) / base_name.format(job.algorithm, args.algorithm_suffix, args.n_samples, "ptheta", job.run_no), prop_theta)
    np.save(Path(experiment._folder_raw) / base_name.format(job.algorithm, args.algorithm_suffix, args.n_samples, "ar", job.run_no), ar)

    total_runtime = posterior_sample_generation_seconds or 0.0
    total_runtime += td_fit_or_train_seconds or 0.0
    total_runtime += sampling_elapsed

    return summarise_run(
        job,
        args,
        runtime_seconds=total_runtime,
        resumed=False,
        posterior_sample_generation_seconds=posterior_sample_generation_seconds,
        td_proposal_fit_or_train_seconds=td_fit_or_train_seconds,
        rjmcmc_sampling_seconds=sampling_elapsed,
    )


def main():
    args = parse_args()
    configure_flow_training(
        args.flow_device,
        flow_num_samples=args.flow_num_samples,
        flow_hidden_layer_size=args.flow_hidden_layer_size,
        flow_annealing=args.flow_annealing,
    )
    run_device = "cuda" if resolve_flow_training_device(args.flow_device).startswith("cuda") else "cpu"

    algorithms = list(DEFAULT_DATA_DICT["SAS"].keys())
    jobs = [Job(algorithm=algorithm, run_no=run_no) for run_no in range(args.start, args.end + 1) for algorithm in algorithms]

    experiments = Experiments(start=args.start, end=args.end, problems=["SAS"], algorithms=algorithms, device=run_device)
    summary_csv = Path(args.summary_csv)
    existing_rows = load_existing_summary(summary_csv)
    row_lookup = {}

    for job in jobs:
        theta_file = raw_path(job.algorithm, args.algorithm_suffix, args.n_samples, "theta", job.run_no)
        ptheta_file = raw_path(job.algorithm, args.algorithm_suffix, args.n_samples, "ptheta", job.run_no)
        ar_file = raw_path(job.algorithm, args.algorithm_suffix, args.n_samples, "ar", job.run_no)

        if args.resume and theta_file.exists() and ptheta_file.exists() and ar_file.exists():
            print(f"Skipping existing run: {job.algorithm} Exp{job.run_no}")
            existing_row = existing_rows.get(job_key(job))
            if existing_row is not None:
                row_lookup[job_key(job)] = existing_row
            else:
                row_lookup[job_key(job)] = summarise_run(job, args, runtime_seconds=None, resumed=True)
        else:
            seed = int(experiments.seeds[0, job.run_no - 1])
            print(f"Running {job.algorithm} Exp{job.run_no}")
            row_lookup[job_key(job)] = run_job(job, args, seed, run_device)

        rows = [row_lookup[job_key(item)] for item in jobs if job_key(item) in row_lookup]
        write_summary(rows, summary_csv)
        write_runtime_summary(rows, Path(args.runtime_summary_csv))

    print(f"Wrote summary to {args.summary_csv}")
    print(f"Wrote runtime summary to {args.runtime_summary_csv}")


if __name__ == "__main__":
    main()
