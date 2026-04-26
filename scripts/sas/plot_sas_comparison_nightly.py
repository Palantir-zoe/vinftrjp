import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from scripts.sas.sas_base import DEFAULT_DATA_DICT, N_SAMPLES, configure_flow_training
from scripts.sas.sas_vinfs import get_normalizing_flows
from src.tools import PlotSinharcsinhPropAll
from src.utils.tools import kde_1D, kde_joint

GROUND_TRUTH_MODEL2 = 0.75
K_COLUMN = 1


def _parse_bool(value):
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot SAS multi-run comparisons across current algorithms.")
    parser.add_argument("--start", type=int, default=1, help="First experiment index included in plots.")
    parser.add_argument("--end", type=int, default=10, help="Last experiment index included in plots.")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES, help="Number of RJMCMC samples per run.")
    parser.add_argument(
        "--algorithm-suffix",
        type=str,
        default="gpuanneal_",
        help="Suffix used in the SAS result filenames.",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="docs/figures/sas_gpuanneal",
        help="Directory to save the generated figures.",
    )
    parser.add_argument(
        "--proposal-run-no",
        type=int,
        default=1,
        help="Experiment index used for the SAS proposal visualization.",
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
        "--proposal-points",
        type=int,
        default=100000,
        help="Calibration sample count used when rendering the proposal figure.",
    )
    return parser.parse_args()


def raw_path(algorithm, suffix, n_samples, kind, run_no):
    return Path("data") / "raw" / f"SAS_{algorithm}_{suffix}NS{n_samples}_{kind}_Exp{run_no}.npy"


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


def load_runs(args):
    algorithms = list(DEFAULT_DATA_DICT["SAS"].keys())
    runs = {algorithm: [] for algorithm in algorithms}
    for algorithm in algorithms:
        for run_no in range(args.start, args.end + 1):
            theta = np.load(raw_path(algorithm, args.algorithm_suffix, args.n_samples, "theta", run_no))
            prop_theta = np.load(raw_path(algorithm, args.algorithm_suffix, args.n_samples, "ptheta", run_no))
            runs[algorithm].append(
                {
                    "run_no": run_no,
                    "theta": theta,
                    "prop_theta": prop_theta,
                    "indicator": theta[:, K_COLUMN],
                    "accepted_move_rate": accepted_move_rate(theta, prop_theta),
                    "ess": effective_sample_size_indicator(theta[:, K_COLUMN]),
                }
            )
    return runs


def plot_running_model_probability(args, runs, figures_dir):
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    steps = np.arange(1, args.n_samples + 1)
    plt.close()
    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=(7, 4))

    for algorithm, algo_dict in DEFAULT_DATA_DICT["SAS"].items():
        traces = []
        for run in runs[algorithm]:
            trace = run["indicator"].cumsum() / steps
            traces.append(trace)
            ax.plot(steps, trace, color=algo_dict["color"], linewidth=0.5, alpha=0.18)

        mean_trace = np.mean(np.vstack(traces), axis=0)
        ax.plot(steps, mean_trace, color=algo_dict["color"], linewidth=2.0, alpha=0.95, label=algo_dict["title"])

    ax.axhline(GROUND_TRUTH_MODEL2, color="black", linewidth=1.2, linestyle="--", label="Ground Truth")
    ax.set_ylabel(r"$\Pr(k=2 \mid \cdot)$ Running Estimate")
    ax.set_xlabel("RJMCMC Step")
    ax.set_ylim((0.55, 0.85))
    ax.set_title(f"SAS RJMCMC Comparison, Runs {args.start}-{args.end}")
    ax.legend(loc="lower right", frameon=True, framealpha=0.8)
    plt.tight_layout()

    suffix = f"runs{args.start}-{args.end}"
    pdf_path = figures_dir / f"SAS_running_mp_trace_{args.algorithm_suffix}NS{args.n_samples}_{suffix}.pdf"
    png_path = figures_dir / f"SAS_running_mp_trace_{args.algorithm_suffix}NS{args.n_samples}_{suffix}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    return pdf_path, png_path


def plot_acceptance_ess(args, runs, figures_dir):
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    plt.close()
    plt.style.use("seaborn-v0_8-white")
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    algorithms = list(DEFAULT_DATA_DICT["SAS"].keys())
    labels = [DEFAULT_DATA_DICT["SAS"][algorithm]["title"] for algorithm in algorithms]
    colors = [DEFAULT_DATA_DICT["SAS"][algorithm]["color"] for algorithm in algorithms]

    acceptance_values = [[run["accepted_move_rate"] for run in runs[algorithm]] for algorithm in algorithms]
    ess_values = [[run["ess"] for run in runs[algorithm]] for algorithm in algorithms]

    for ax, values, title, ylabel in [
        (axes[0], acceptance_values, "Acceptance Comparison", "Accepted-Move Rate"),
        (axes[1], ess_values, "Model-Indicator ESS Comparison", r"ESS of $1\{k=2\}$"),
    ]:
        box = ax.boxplot(values, patch_artist=True, labels=labels)
        for patch, color in zip(box["boxes"], colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.18)
            patch.set_edgecolor(color)
            patch.set_linewidth(1.5)
        for median, color in zip(box["medians"], colors, strict=True):
            median.set_color(color)
            median.set_linewidth(2)
        for i, vals in enumerate(values, start=1):
            jitter = np.linspace(-0.08, 0.08, num=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, color=colors[i - 1], alpha=0.75, s=36)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    suffix = f"runs{args.start}-{args.end}"
    pdf_path = figures_dir / f"SAS_acceptance_ess_{args.algorithm_suffix}NS{args.n_samples}_{suffix}.pdf"
    png_path = figures_dir / f"SAS_acceptance_ess_{args.algorithm_suffix}NS{args.n_samples}_{suffix}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    return pdf_path, png_path


def plot_proposals(args, figures_dir):
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    plt.close()
    plt.style.use("seaborn-v0_8-white")

    prob_dict = DEFAULT_DATA_DICT["SAS"]
    nd = 30
    plotter = PlotSinharcsinhPropAll("SAS", seed=args.proposal_run_no, nd=nd, n_samples=args.proposal_points)
    m1theta, m2theta, p_sas_1d, p_u, pt_prop_m1theta = plotter.run(
        list(prob_dict.keys()),
        verbose=False,
        normalizing_flows=get_normalizing_flows,
        save_flows_dir=args.save_flows_dir,
    )

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 6))
    ax = ax.flatten()

    ax[0].set_title("Source 1D SAS")
    kde_1D(ax[0], m1theta[:, 0], bw=1, plotline=False)
    ax[0].scatter(p_sas_1d, np.zeros_like(p_u), c=p_u, marker="x", cmap="brg")
    ax[0].set_yticklabels([])

    for i, algorithm in enumerate(prob_dict):
        ax[i + 1].set_title(prob_dict[algorithm]["title"])
        kde_joint(ax[i + 1], m2theta[:, [0, 2]], cmap="Blues", alpha=1, bw=0.1, maxz_scale=2, n_grid_points=128)
        ax[i + 1].scatter(
            pt_prop_m1theta[algorithm][:, 0],
            pt_prop_m1theta[algorithm][:, 2],
            s=0.35,
            alpha=0.45,
            c=np.tile(p_u, nd),
            cmap="brg",
        )
        ax[i + 1].set_xlim([-1, 15])
        ax[i + 1].set_ylim([-7, 0.5])

    plt.tight_layout()
    suffix = f"run{args.proposal_run_no}"
    pdf_path = figures_dir / f"SAS_proposal_{args.algorithm_suffix}NS{args.proposal_points}_{suffix}.pdf"
    png_path = figures_dir / f"SAS_proposal_{args.algorithm_suffix}NS{args.proposal_points}_{suffix}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    return pdf_path, png_path


def main():
    args = parse_args()
    configure_flow_training(
        args.flow_device,
        flow_num_samples=args.flow_num_samples,
        flow_hidden_layer_size=args.flow_hidden_layer_size,
        flow_annealing=args.flow_annealing,
    )

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args)
    running_paths = plot_running_model_probability(args, runs, figures_dir)
    acceptance_paths = plot_acceptance_ess(args, runs, figures_dir)
    proposal_paths = plot_proposals(args, figures_dir)

    for algorithm, algo_dict in DEFAULT_DATA_DICT["SAS"].items():
        acceptance_mean = np.mean([run["accepted_move_rate"] for run in runs[algorithm]])
        ess_mean = np.mean([run["ess"] for run in runs[algorithm]])
        prob_mean = np.mean([np.mean(run["indicator"]) for run in runs[algorithm]])
        print(
            f"{algo_dict['title']}: mean accepted-move rate={acceptance_mean:.4f}, "
            f"mean ESS={ess_mean:.1f}, mean Pr(k=2)={prob_mean:.4f}"
        )

    print(f"Saved running trace plots to {running_paths[0]} and {running_paths[1]}")
    print(f"Saved acceptance/ESS plots to {acceptance_paths[0]} and {acceptance_paths[1]}")
    print(f"Saved proposal plots to {proposal_paths[0]} and {proposal_paths[1]}")


if __name__ == "__main__":
    main()
