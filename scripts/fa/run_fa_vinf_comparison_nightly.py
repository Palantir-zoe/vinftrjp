import argparse
import contextlib
import csv
import gc
import json
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from scripts.fa.fa_base import N_SAMPLES, configure_flow_training, get_train_theta_start_theta
from scripts.fa.fa_vinfs import get_normalizing_flows
from src.algorithms import FactorAnalysisModelVINF, FactorAnalysisModelVINFIS
from src.problems import FA
from src.samplers import RJMCMC


@dataclass(frozen=True)
class Job:
    algorithm: str
    run_no: int


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overnight FA comparison between vanilla VINF and VINFIS at a fixed particle count."
    )
    parser.add_argument("--run-start", type=int, default=1, help="First FA run index to execute.")
    parser.add_argument("--run-end", type=int, default=10, help="Last FA run index to execute.")
    parser.add_argument("--n-particles", type=int, default=8000, help="Calibration particle count.")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES, help="Number of RJMCMC iterations per run.")
    parser.add_argument(
        "--importance-samples",
        type=int,
        default=4000,
        help="Number of flow samples per model used by VINFIS to estimate posterior model probabilities.",
    )
    parser.add_argument(
        "--flow-device",
        type=str,
        default="cuda",
        help="Device for flow training only, e.g. 'cpu', 'cuda', or 'auto'. Sampling remains on CPU.",
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
    parser.add_argument(
        "--save-flows-dir",
        type=str,
        default="data/flows/fa_nightly_n8000",
        help="Directory used to cache trained flows across overnight runs.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw",
        help="Directory containing the precomputed FA calibration draws.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/nightly_fa_vinf_vs_vinfis_n8000",
        help="Directory used to save overnight chain outputs and summaries.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="nightly",
        help="Tag inserted into output filenames so overnight results stay separate from earlier runs.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip jobs whose full output set already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate inputs and print the planned jobs without running sampling.",
    )
    return parser.parse_args()


def set_random_seed(seed: int, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float32)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if hasattr(torch, "use_deterministic_algorithms"):
                torch.use_deterministic_algorithms(True)


def get_seed_for_run(run_no: int) -> int:
    seeds = np.random.default_rng(2025).integers(np.iinfo(np.int32).max, size=1000)
    return int(seeds[run_no - 1])


def get_vinfis_proposal(proposal):
    if hasattr(proposal, "posterior_model_probabilities"):
        return proposal

    for subproposal in getattr(proposal, "ps", []):
        if hasattr(subproposal, "posterior_model_probabilities"):
            return subproposal

    return proposal


def build_jobs(run_start: int, run_end: int):
    jobs = []
    for run_no in range(run_start, run_end + 1):
        jobs.append(Job("FactorAnalysisModelVINF", run_no))
        jobs.append(Job("FactorAnalysisModelVINFIS", run_no))
    return jobs


def expected_input_paths(input_dir: Path, n_particles: int, run_no: int):
    paths = []
    for k in (1, 2):
        paths.append(input_dir / f"FA_pyMC_k{k}_N{n_particles}_run{run_no}_train.npy")
        paths.append(input_dir / f"FA_pyMC_k{k}_N{n_particles}_run{run_no}_test.npy")
    return paths


def make_output_stem(job: Job, args):
    return f"FA_{job.algorithm}_N{args.n_particles}_pyMC_run{job.run_no}_{args.tag}_NS{args.n_samples}"


def expected_output_paths(output_dir: Path, stem: str):
    return [
        output_dir / f"{stem}_theta.npy",
        output_dir / f"{stem}_ptheta.npy",
        output_dir / f"{stem}_ar.npy",
        output_dir / f"{stem}_llh.npy",
        output_dir / f"{stem}_logprior.npy",
        output_dir / f"{stem}_summary.json",
    ]


def save_summary_csv(summary_path: Path, rows):
    fieldnames = [
        "algorithm",
        "run_no",
        "status",
        "seed",
        "runtime_seconds",
        "acceptance_rate",
        "chain_prob_k1",
        "chain_prob_k2",
        "estimated_prob_k1",
        "estimated_prob_k2",
        "importance_samples",
        "output_stem",
        "log_path",
        "error",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_job(job: Job, args, y_data, output_dir: Path):
    seed = get_seed_for_run(job.run_no)
    set_random_seed(seed, deterministic=True)

    train_theta, start_theta = get_train_theta_start_theta(Path(args.input_dir), job.run_no, "FA", args.n_particles)
    problem = FA()

    common_kwargs = {
        "problem": problem,
        "y_data": y_data,
        "normalizing_flows": get_normalizing_flows,
        "save_flows_dir": args.save_flows_dir,
    }
    if job.algorithm == "FactorAnalysisModelVINF":
        model = FactorAnalysisModelVINF(**common_kwargs)
    elif job.algorithm == "FactorAnalysisModelVINFIS":
        model = FactorAnalysisModelVINFIS(
            **common_kwargs,
            importance_num_samples=args.importance_samples,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {job.algorithm}")

    sampler = RJMCMC(model, calibrate_draws=train_theta)
    started = time.time()
    theta, prop_theta, llh, log_prior, ar = sampler.run(args.n_samples, start_theta=start_theta)
    runtime_seconds = time.time() - started

    stem = make_output_stem(job, args)
    np.save(output_dir / f"{stem}_theta.npy", theta)
    np.save(output_dir / f"{stem}_ptheta.npy", prop_theta)
    np.save(output_dir / f"{stem}_ar.npy", ar)
    np.save(output_dir / f"{stem}_llh.npy", llh)
    np.save(output_dir / f"{stem}_logprior.npy", log_prior)

    k_col = model.generateRVIndices()["k"][0]
    k_chain = np.asarray(theta[:, k_col], dtype=np.int64)
    chain_prob_k1 = float(np.mean(k_chain == 1))
    chain_prob_k2 = float(np.mean(k_chain == 2))
    acceptance_rate = float(np.mean(ar))

    summary = {
        "algorithm": job.algorithm,
        "run_no": job.run_no,
        "seed": seed,
        "n_particles": args.n_particles,
        "n_samples": args.n_samples,
        "runtime_seconds": runtime_seconds,
        "acceptance_rate": acceptance_rate,
        "chain_prob_k1": chain_prob_k1,
        "chain_prob_k2": chain_prob_k2,
        "flow_device": args.flow_device,
        "flow_num_samples": args.flow_num_samples,
        "flow_hidden_layer_size": args.flow_hidden_layer_size,
        "save_flows_dir": args.save_flows_dir,
    }

    if job.algorithm == "FactorAnalysisModelVINFIS":
        proposal = get_vinfis_proposal(model.proposal)
        estimated_probs = getattr(proposal, "posterior_model_probabilities", {}) or {}
        estimated_log_marginals = getattr(proposal, "estimated_log_marginal_likelihoods", {}) or {}
        summary["importance_samples"] = args.importance_samples
        summary["estimated_probabilities"] = {str(mk): float(prob) for mk, prob in estimated_probs.items()}
        summary["estimated_log_marginal_likelihoods"] = {
            str(mk): float(log_marginal) for mk, log_marginal in estimated_log_marginals.items()
        }

    summary_path = output_dir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "algorithm": job.algorithm,
        "run_no": job.run_no,
        "status": "completed",
        "seed": seed,
        "runtime_seconds": f"{runtime_seconds:.2f}",
        "acceptance_rate": f"{acceptance_rate:.6f}",
        "chain_prob_k1": f"{chain_prob_k1:.6f}",
        "chain_prob_k2": f"{chain_prob_k2:.6f}",
        "estimated_prob_k1": (
            f"{summary.get('estimated_probabilities', {}).get('(1,)', float('nan')):.6f}"
            if job.algorithm == "FactorAnalysisModelVINFIS"
            else ""
        ),
        "estimated_prob_k2": (
            f"{summary.get('estimated_probabilities', {}).get('(2,)', float('nan')):.6f}"
            if job.algorithm == "FactorAnalysisModelVINFIS"
            else ""
        ),
        "importance_samples": args.importance_samples if job.algorithm == "FactorAnalysisModelVINFIS" else "",
        "output_stem": stem,
        "log_path": "",
        "error": "",
    }


def main():
    args = parse_args()
    if args.run_start <= 0 or args.run_end < args.run_start:
        raise ValueError(f"Invalid run range: start={args.run_start}, end={args.run_end}")

    configure_flow_training(args.flow_device, args.flow_num_samples, args.flow_hidden_layer_size)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    logs_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    missing_inputs = []
    for run_no in range(args.run_start, args.run_end + 1):
        for path in expected_input_paths(input_dir, args.n_particles, run_no):
            if not path.exists():
                missing_inputs.append(str(path))
    if missing_inputs:
        missing = "\n".join(missing_inputs[:10])
        raise FileNotFoundError(f"Missing FA calibration inputs for the requested overnight run range:\n{missing}")

    jobs = build_jobs(args.run_start, args.run_end)
    summary_rows = []
    summary_csv_path = output_dir / "overnight_summary.csv"

    print(f"Prepared {len(jobs)} jobs.")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Flow cache dir: {args.save_flows_dir}")
    for job in jobs:
        print(f"  - run {job.run_no:02d}: {job.algorithm}")

    if args.dry_run:
        return

    y_data = np.load(str(Path("data") / "core" / "FA_data.npy"))

    for job in jobs:
        stem = make_output_stem(job, args)
        outputs = expected_output_paths(output_dir, stem)
        log_path = logs_dir / f"{stem}.log"

        if args.resume and all(path.exists() for path in outputs):
            row = {
                "algorithm": job.algorithm,
                "run_no": job.run_no,
                "status": "skipped_existing",
                "seed": get_seed_for_run(job.run_no),
                "runtime_seconds": "",
                "acceptance_rate": "",
                "chain_prob_k1": "",
                "chain_prob_k2": "",
                "estimated_prob_k1": "",
                "estimated_prob_k2": "",
                "importance_samples": args.importance_samples if job.algorithm == "FactorAnalysisModelVINFIS" else "",
                "output_stem": stem,
                "log_path": str(log_path),
                "error": "",
            }
            summary_rows.append(row)
            save_summary_csv(summary_csv_path, summary_rows)
            print(f"Skipping completed job: run {job.run_no:02d} {job.algorithm}")
            continue

        print(f"Starting job: run {job.run_no:02d} {job.algorithm}")
        with log_path.open("w", encoding="utf-8") as log_handle:
            tee = TeeStream(sys.stdout, log_handle)
            try:
                with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                    print("=" * 80)
                    print(f"Job start: run={job.run_no}, algorithm={job.algorithm}, seed={get_seed_for_run(job.run_no)}")
                    print(f"n_particles={args.n_particles}, n_samples={args.n_samples}, flow_device={args.flow_device}")
                    print(f"flow_num_samples={args.flow_num_samples}, flow_hidden_layer_size={args.flow_hidden_layer_size}")
                    print(f"save_flows_dir={args.save_flows_dir}")
                    row = run_job(job, args, y_data, output_dir)
                    print(f"Job completed: run={job.run_no}, algorithm={job.algorithm}")
            except Exception:
                error_text = traceback.format_exc()
                row = {
                    "algorithm": job.algorithm,
                    "run_no": job.run_no,
                    "status": "failed",
                    "seed": get_seed_for_run(job.run_no),
                    "runtime_seconds": "",
                    "acceptance_rate": "",
                    "chain_prob_k1": "",
                    "chain_prob_k2": "",
                    "estimated_prob_k1": "",
                    "estimated_prob_k2": "",
                    "importance_samples": args.importance_samples if job.algorithm == "FactorAnalysisModelVINFIS" else "",
                    "output_stem": stem,
                    "log_path": str(log_path),
                    "error": error_text.strip().splitlines()[-1],
                }
                with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                    print(error_text)
                    print(f"Job failed: run={job.run_no}, algorithm={job.algorithm}")
            row["log_path"] = str(log_path)
            summary_rows.append(row)
            save_summary_csv(summary_csv_path, summary_rows)


if __name__ == "__main__":
    main()
