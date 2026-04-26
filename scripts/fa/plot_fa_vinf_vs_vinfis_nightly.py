import argparse
import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from scripts.fa.fa_base import configure_flow_training, get_train_theta_start_theta
from scripts.fa.fa_vinfs import get_normalizing_flows
from scripts.fa.plot_fa_proposal import RJPropTest
from src.algorithms import FactorAnalysisModelVINF, FactorAnalysisModelVINFIS
from src.problems import FA
from src.utils.tools import kde_joint

GROUND_TRUTH_K1 = 0.882
K_COLUMN = 15
ALGORITHM_SPECS = [
    ("FactorAnalysisModelVINF", "#1f77b4", "VINF"),
    ("FactorAnalysisModelVINFIS", "#d62728", "VINFIS"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot nightly FA results for vanilla VINF versus VINFIS, including proposal and RJMCMC figures."
    )
    parser.add_argument(
        "--nightly-output-dir",
        type=str,
        default="data/raw/nightly_fa_vinf_vs_vinfis_n8000",
        help="Directory containing the nightly VINF and VINFIS chain outputs.",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="docs/figures/nightly_fa_vinf_vs_vinfis_n8000",
        help="Directory where the generated figures will be written.",
    )
    parser.add_argument(
        "--runtime-summary-csv",
        type=str,
        default="",
        help="Optional path to the runtime summary CSV. Defaults to <nightly-output-dir>/overnight_summary.csv.",
    )
    parser.add_argument("--run-start", type=int, default=1, help="First nightly run index to include in the chain plot.")
    parser.add_argument("--run-end", type=int, default=10, help="Last nightly run index to include in the chain plot.")
    parser.add_argument(
        "--proposal-run-no",
        type=int,
        default=1,
        help="Run index whose calibration draws are used to construct the proposal comparison figure.",
    )
    parser.add_argument("--n-particles", type=int, default=8000, help="Particle count used by the nightly comparison.")
    parser.add_argument("--n-samples", type=int, default=100000, help="Number of RJMCMC steps in each saved chain.")
    parser.add_argument("--tag", type=str, default="nightly", help="Filename tag used by the nightly outputs.")
    parser.add_argument(
        "--flow-device",
        type=str,
        default="cuda",
        help="Device for proposal-figure flow training/loading only, e.g. 'cpu', 'cuda', or 'auto'.",
    )
    parser.add_argument(
        "--flow-num-samples",
        type=int,
        default=None,
        help="Override the number of Monte Carlo samples per flow-training iteration for the proposal plot.",
    )
    parser.add_argument(
        "--flow-hidden-layer-size",
        type=int,
        default=None,
        help="Override the hidden width used by FA flow training networks for the proposal plot.",
    )
    parser.add_argument(
        "--save-flows-dir",
        type=str,
        default="data/flows/fa_nightly_n8000",
        help="Flow cache directory reused when constructing the proposal figure.",
    )
    parser.add_argument(
        "--importance-samples",
        type=int,
        default=4000,
        help="Number of flow samples per model used by VINFIS during proposal calibration.",
    )
    parser.add_argument(
        "--proposal-points",
        type=int,
        default=200,
        help="Number of source points shown in the proposal figure.",
    )
    parser.add_argument(
        "--proposal-grid",
        type=int,
        default=256,
        help="Grid size used in KDE contour plots for the proposal figure.",
    )
    return parser.parse_args()


def setup_matplotlib():
    plt.close("all")
    plt.style.use("seaborn-v0_8-white")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42


def make_chain_file(output_dir: Path, algorithm: str, run_no: int, tag: str, n_particles: int, n_samples: int):
    stem = f"FA_{algorithm}_N{n_particles}_pyMC_run{run_no}_{tag}_NS{n_samples}"
    return output_dir / f"{stem}_theta.npy"


def make_prop_file(output_dir: Path, algorithm: str, run_no: int, tag: str, n_particles: int, n_samples: int):
    stem = f"FA_{algorithm}_N{n_particles}_pyMC_run{run_no}_{tag}_NS{n_samples}"
    return output_dir / f"{stem}_ptheta.npy"


def load_chain_probabilities(output_dir: Path, algorithm: str, run_start: int, run_end: int, tag: str, n_particles: int, n_samples: int):
    chains = []
    included_runs = []
    for run_no in range(run_start, run_end + 1):
        path = make_chain_file(output_dir, algorithm, run_no, tag, n_particles, n_samples)
        if not path.exists():
            continue
        theta = np.load(str(path))
        if theta.ndim == 1:
            theta = theta[None, :]
        k = theta[:, K_COLUMN]
        chains.append((k == 1).cumsum() / np.arange(1, k.shape[0] + 1))
        included_runs.append(run_no)
    return chains, included_runs


def accepted_move_rate(theta, prop_theta):
    theta = np.asarray(theta)
    prop_theta = np.asarray(prop_theta)
    if theta.shape != prop_theta.shape or theta.shape[0] <= 1:
        return float("nan")
    accepted = np.any(np.abs(theta[1:] - prop_theta[1:]) > 1e-12, axis=1)
    return float(np.mean(accepted))


def effective_sample_size_indicator(samples):
    x = np.asarray(samples, dtype=np.float64)
    n = x.shape[0]
    if n < 4:
        return float(n)
    x = x - x.mean()
    var = np.dot(x, x) / n
    if not np.isfinite(var) or var <= 0.0:
        return 0.0

    fft_size = 1 << (2 * n - 1).bit_length()
    centered_fft = np.fft.rfft(x, n=fft_size)
    acov = np.fft.irfft(centered_fft * np.conjugate(centered_fft), n=fft_size)[:n]
    acov /= np.arange(n, 0, -1)
    rho = acov / acov[0]

    tau = 1.0
    for lag in range(1, n - 1, 2):
        pair_sum = rho[lag] + rho[lag + 1]
        if not np.isfinite(pair_sum) or pair_sum <= 0.0:
            break
        tau += 2.0 * pair_sum

    ess = n / tau
    return float(max(1.0, min(n, ess)))


def load_acceptance_and_ess(output_dir: Path, algorithm: str, run_start: int, run_end: int, tag: str, n_particles: int, n_samples: int):
    rows = []
    for run_no in range(run_start, run_end + 1):
        theta_path = make_chain_file(output_dir, algorithm, run_no, tag, n_particles, n_samples)
        prop_path = make_prop_file(output_dir, algorithm, run_no, tag, n_particles, n_samples)
        if not theta_path.exists() or not prop_path.exists():
            continue

        theta = np.load(str(theta_path))
        prop_theta = np.load(str(prop_path))
        if theta.ndim == 1:
            theta = theta[None, :]
        if prop_theta.ndim == 1:
            prop_theta = prop_theta[None, :]

        indicator = (theta[:, K_COLUMN] == 1).astype(np.float64)
        rows.append(
            {
                "run_no": run_no,
                "acceptance_rate": accepted_move_rate(theta, prop_theta),
                "ess_indicator_k1": effective_sample_size_indicator(indicator),
            }
        )
    return rows


def load_runtime_rows(summary_csv: Path, run_start: int, run_end: int):
    if not summary_csv.exists():
        return []

    rows = []
    with summary_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            run_no = int(row["run_no"])
            if not (run_start <= run_no <= run_end):
                continue
            if row.get("status") not in {"completed", "skipped_existing"}:
                continue
            rows.append(row)
    return rows


def plot_runtime_comparison(summary_csv: Path, figures_dir: Path, args):
    setup_matplotlib()
    rows = load_runtime_rows(summary_csv, args.run_start, args.run_end)
    if not rows:
        return None, None, {}

    metrics = {}
    for algorithm, _, label in ALGORITHM_SPECS:
        algorithm_rows = [row for row in rows if row["algorithm"] == algorithm]
        metrics[label] = {
            "td_proposal_fit_or_train_seconds": [
                float(row["td_proposal_fit_or_train_seconds"])
                for row in algorithm_rows
                if row.get("td_proposal_fit_or_train_seconds", "") not in {"", None}
            ],
            "rjmcmc_sampling_seconds": [
                float(row["rjmcmc_sampling_seconds"])
                for row in algorithm_rows
                if row.get("rjmcmc_sampling_seconds", "") not in {"", None}
            ],
            "runtime_seconds": [
                float(row["runtime_seconds"]) for row in algorithm_rows if row.get("runtime_seconds", "") not in {"", None}
            ],
        }

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(11, 3.8))
    x_positions = np.arange(len(ALGORITHM_SPECS))
    plot_specs = [
        ("td_proposal_fit_or_train_seconds", "TD Proposal Time (s)", "TD Proposal Fit/Train"),
        ("rjmcmc_sampling_seconds", "Sampling Time (s)", "RJMCMC Sampling"),
        ("runtime_seconds", "Method Runtime (s)", "Method Runtime"),
    ]

    for ax, (field_name, ylabel, title) in zip(axes, plot_specs, strict=True):
        values = [metrics[label][field_name] for _, _, label in ALGORITHM_SPECS]
        bp = ax.boxplot(values, positions=x_positions, widths=0.55, patch_artist=True, showfliers=False)
        for patch, (_, color, _) in zip(bp["boxes"], ALGORITHM_SPECS, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.18)
            patch.set_edgecolor(color)
        for median, (_, color, _) in zip(bp["medians"], ALGORITHM_SPECS, strict=True):
            median.set_color(color)
            median.set_linewidth(1.6)

        for xpos, run_values, (_, color, _) in zip(x_positions, values, ALGORITHM_SPECS, strict=True):
            jitter = np.linspace(-0.08, 0.08, len(run_values)) if run_values else np.array([])
            ax.scatter(np.full(len(run_values), xpos) + jitter, run_values, color=color, s=24, alpha=0.8)

        ax.set_xticks(x_positions, [label for _, _, label in ALGORITHM_SPECS])
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    plt.tight_layout()
    suffix = f"runs{args.run_start}-{args.run_end}_{args.tag}"
    pdf_path = figures_dir / f"FA_runtime_VINF_vs_VINFIS_N{args.n_particles}_{suffix}.pdf"
    png_path = figures_dir / f"FA_runtime_VINF_vs_VINFIS_N{args.n_particles}_{suffix}.png"
    fig.savefig(str(pdf_path))
    fig.savefig(str(png_path), dpi=200)
    plt.close(fig)
    return pdf_path, png_path, metrics


def plot_acceptance_and_ess(output_dir: Path, figures_dir: Path, args):
    setup_matplotlib()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8.6, 3.8))
    metrics = {}

    for algorithm, color, label in ALGORITHM_SPECS:
        metrics[label] = load_acceptance_and_ess(
            output_dir, algorithm, args.run_start, args.run_end, args.tag, args.n_particles, args.n_samples
        )

    x_positions = np.arange(len(ALGORITHM_SPECS))
    acceptance_values = [[row["acceptance_rate"] for row in metrics[label]] for _, _, label in ALGORITHM_SPECS]
    ess_values = [[row["ess_indicator_k1"] for row in metrics[label]] for _, _, label in ALGORITHM_SPECS]

    for ax, values, ylabel, title in [
        (axes[0], acceptance_values, "Accepted-Move Rate", "Acceptance Comparison"),
        (axes[1], ess_values, r"ESS of $1\{k=1\}$", "Model-Indicator ESS Comparison"),
    ]:
        bp = ax.boxplot(values, positions=x_positions, widths=0.55, patch_artist=True, showfliers=False)
        for patch, (_, color, _) in zip(bp["boxes"], ALGORITHM_SPECS, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.18)
            patch.set_edgecolor(color)
        for median, (_, color, _) in zip(bp["medians"], ALGORITHM_SPECS, strict=True):
            median.set_color(color)
            median.set_linewidth(1.6)

        for xpos, run_values, (_, color, _) in zip(x_positions, values, ALGORITHM_SPECS, strict=True):
            jitter = np.linspace(-0.08, 0.08, len(run_values)) if run_values else np.array([])
            ax.scatter(np.full(len(run_values), xpos) + jitter, run_values, color=color, s=24, alpha=0.8)

        ax.set_xticks(x_positions, [label for _, _, label in ALGORITHM_SPECS])
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    plt.tight_layout()
    suffix = f"runs{args.run_start}-{args.run_end}_{args.tag}"
    pdf_path = figures_dir / f"FA_acceptance_ess_VINF_vs_VINFIS_N{args.n_particles}_{suffix}.pdf"
    png_path = figures_dir / f"FA_acceptance_ess_VINF_vs_VINFIS_N{args.n_particles}_{suffix}.png"
    fig.savefig(str(pdf_path))
    fig.savefig(str(png_path), dpi=200)
    plt.close(fig)
    return pdf_path, png_path, metrics


def beta_label(name: str):
    row = int(name[4])
    col = int(name[5])
    return rf"$\beta_{{{col + 1},{row + 1}}}$"


def proposal_projections():
    return [
        {
            "source": ("beta31", "beta41"),
            "target": ("beta31", "beta32"),
            "title_suffix": "Obs 4",
        },
        {
            "source": ("beta41", "beta51"),
            "target": ("beta41", "beta42"),
            "title_suffix": "Obs 5",
        },
        {
            "source": ("beta31", "beta51"),
            "target": ("beta51", "beta52"),
            "title_suffix": "Obs 6",
        },
    ]


def get_column_map(model):
    columns = model.generateRVIndices()
    return {name: int(columns[name][0]) for name in columns if name.startswith("beta")}


def plot_rjmcmc(output_dir: Path, figures_dir: Path, args):
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 4))

    available = {}

    for algorithm, color, label in ALGORITHM_SPECS:
        chains, included_runs = load_chain_probabilities(
            output_dir, algorithm, args.run_start, args.run_end, args.tag, args.n_particles, args.n_samples
        )
        if not chains:
            continue

        stack = np.stack(chains, axis=0)
        steps = np.arange(1, stack.shape[1] + 1)
        mean = stack.mean(axis=0)
        std = stack.std(axis=0)
        for chain in stack:
            ax.plot(steps, chain, color=color, linewidth=0.8, alpha=0.18)
        ax.plot(steps, mean, color=color, linewidth=2.0, label=f"{label} mean")
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.12)
        available[label] = included_runs

    ax.axhline(GROUND_TRUTH_K1, color="black", linestyle="--", linewidth=1.2, label="Ground truth")
    ax.set_xlabel("RJMCMC Step")
    ax.set_ylabel("2-Factor Model Probability Estimate")
    ax.set_title(f"FA RJMCMC at N={args.n_particles}: VINF vs VINFIS")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right", frameon=True, framealpha=0.8)
    plt.tight_layout()

    suffix = f"runs{args.run_start}-{args.run_end}_{args.tag}"
    pdf_path = figures_dir / f"FA_rjmcmc_VINF_vs_VINFIS_N{args.n_particles}_{suffix}.pdf"
    png_path = figures_dir / f"FA_rjmcmc_VINF_vs_VINFIS_N{args.n_particles}_{suffix}.png"
    fig.savefig(str(pdf_path))
    fig.savefig(str(png_path), dpi=200)
    plt.close(fig)
    return pdf_path, png_path, available


def load_gold_draws():
    data_folder = Path("data") / "core"
    mk_theta = {}
    mk_theta_mode = {}
    for k in range(2, 4):
        draws = np.load(str(data_folder / f"gold_m{k}.npy"))
        mk_theta[k] = draws
        mk_theta_mode[k] = draws[0::100]
    return mk_theta[2], mk_theta[3], mk_theta_mode


def build_model(algorithm: str, y_data, save_flows_dir: str, importance_samples: int):
    common_kwargs = {
        "problem": FA(),
        "y_data": y_data,
        "normalizing_flows": get_normalizing_flows,
        "save_flows_dir": save_flows_dir,
    }
    if algorithm == "FactorAnalysisModelVINF":
        return FactorAnalysisModelVINF(**common_kwargs)
    if algorithm == "FactorAnalysisModelVINFIS":
        return FactorAnalysisModelVINFIS(**common_kwargs, importance_num_samples=importance_samples)
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def plot_proposals(figures_dir: Path, args):
    setup_matplotlib()
    configure_flow_training(args.flow_device, args.flow_num_samples, args.flow_hidden_layer_size)

    y_data = np.load(str(Path("data") / "core" / "FA_data.npy"))
    _, mktt3, mk_theta_mode = load_gold_draws()
    train_theta, _ = get_train_theta_start_theta(Path("data") / "raw", args.proposal_run_no, "FA", args.n_particles)
    source_points = mk_theta_mode[2][: args.proposal_points]

    prop_theta = {}
    estimated_probs = {}
    columns = None
    for algorithm in ("FactorAnalysisModelVINF", "FactorAnalysisModelVINFIS"):
        model = build_model(algorithm, y_data, args.save_flows_dir, args.importance_samples)
        if columns is None:
            columns = get_column_map(model)
        proptest = RJPropTest(model, train_theta[::10])
        prop_theta[algorithm] = proptest.propose(source_points)
        if algorithm == "FactorAnalysisModelVINFIS":
            proposal = model.proposal
            for subproposal in getattr(proposal, "ps", []):
                if hasattr(subproposal, "posterior_model_probabilities"):
                    estimated_probs = subproposal.posterior_model_probabilities or {}
                    break

    projections = proposal_projections()
    fig, axes = plt.subplots(nrows=len(projections), ncols=3, figsize=(11, 10.5))
    axes = axes.flatten()

    for row_idx, projection in enumerate(projections):
        source_x, source_y = [columns[name] for name in projection["source"]]
        target_x, target_y = [columns[name] for name in projection["target"]]

        ax_source = axes[row_idx * 3]
        kde_joint(
            ax_source,
            source_points[:, [source_x, source_y]],
            cmap="Blues",
            alpha=1,
            bw=0.05,
            n_grid_points=args.proposal_grid,
        )
        ax_source.scatter(source_points[:, source_x], source_points[:, source_y], color="#444444", s=4, alpha=0.55)
        ax_source.set_title(f"Source 2-Factor, {projection['title_suffix']}")
        ax_source.set_xlabel(beta_label(projection["source"][0]))
        ax_source.set_ylabel(beta_label(projection["source"][1]))

        for col_offset, (algorithm, color, label) in enumerate(ALGORITHM_SPECS, start=1):
            ax = axes[row_idx * 3 + col_offset]
            kde_joint(
                ax,
                mktt3[:, [target_x, target_y]],
                cmap="Blues",
                alpha=1,
                bw=0.05,
                n_grid_points=args.proposal_grid,
            )
            ax.scatter(prop_theta[algorithm][:, target_x], prop_theta[algorithm][:, target_y], color=color, s=6, alpha=0.65)
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_title(f"Target 3-Factor, {label}, {projection['title_suffix']}")
            ax.set_xlabel(beta_label(projection["target"][0]))
            ax.set_ylabel(beta_label(projection["target"][1]))

    plt.tight_layout()
    suffix = f"N{args.n_particles}_run{args.proposal_run_no}_{args.tag}"
    pdf_path = figures_dir / f"FA_proposal_VINF_vs_VINFIS_{suffix}.pdf"
    png_path = figures_dir / f"FA_proposal_VINF_vs_VINFIS_{suffix}.png"
    fig.savefig(str(pdf_path))
    fig.savefig(str(png_path), dpi=200)
    plt.close(fig)
    return pdf_path, png_path, estimated_probs


def main():
    args = parse_args()
    output_dir = Path(args.nightly_output_dir)
    figures_dir = Path(args.figures_dir)
    runtime_summary_csv = Path(args.runtime_summary_csv) if args.runtime_summary_csv else output_dir / "overnight_summary.csv"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rjmcmc_pdf, rjmcmc_png, available = plot_rjmcmc(output_dir, figures_dir, args)
    acceptance_pdf, acceptance_png, metrics = plot_acceptance_and_ess(output_dir, figures_dir, args)
    runtime_pdf, runtime_png, runtime_metrics = plot_runtime_comparison(runtime_summary_csv, figures_dir, args)
    proposal_pdf, proposal_png, estimated_probs = plot_proposals(figures_dir, args)

    print("Generated FA nightly comparison figures:")
    print(f"  RJMCMC pdf: {rjmcmc_pdf}")
    print(f"  RJMCMC png: {rjmcmc_png}")
    print(f"  Acceptance/ESS pdf: {acceptance_pdf}")
    print(f"  Acceptance/ESS png: {acceptance_png}")
    if runtime_pdf is not None and runtime_png is not None:
        print(f"  Runtime pdf: {runtime_pdf}")
        print(f"  Runtime png: {runtime_png}")
    print(f"  Proposal pdf: {proposal_pdf}")
    print(f"  Proposal png: {proposal_png}")
    if available:
        for label, runs in available.items():
            print(f"  {label} runs included: {runs}")
    if metrics:
        for label, rows in metrics.items():
            if rows:
                mean_acceptance = np.mean([row["acceptance_rate"] for row in rows])
                mean_ess = np.mean([row["ess_indicator_k1"] for row in rows])
                print(f"  {label} mean accepted-move rate: {mean_acceptance:.4f}")
                print(f"  {label} mean ESS(1{{k=1}}): {mean_ess:.1f}")
    if runtime_metrics:
        for label, metric_dict in runtime_metrics.items():
            td_values = metric_dict["td_proposal_fit_or_train_seconds"]
            sampling_values = metric_dict["rjmcmc_sampling_seconds"]
            runtime_values = metric_dict["runtime_seconds"]
            if td_values:
                print(f"  {label} mean TD proposal fit/train time: {np.mean(td_values):.2f}s")
            if sampling_values:
                print(f"  {label} mean RJMCMC sampling time: {np.mean(sampling_values):.2f}s")
            if runtime_values:
                print(f"  {label} mean method runtime: {np.mean(runtime_values):.2f}s")
    if estimated_probs:
        print(f"  VINFIS estimated model probabilities during proposal calibration: {estimated_probs}")


if __name__ == "__main__":
    main()
