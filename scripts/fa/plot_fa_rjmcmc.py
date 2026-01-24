from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from scripts.fa.fa_base import ALGORITHM_COLORS, ALGORITHM_LABELS, N_SAMPLES, NPARTICLES, setup_argparse


def generate_m1problist(
    data_folder, prob: str, algorithms: list[str], n_flows: list[int] | None, n_particles: int, exp: int
):
    args = setup_argparse()

    if n_flows is None:
        n_flows = [0] * len(algorithms)

    m1problist = {}
    for algo, n_flow in zip(algorithms, n_flows, strict=True):
        for run_no in range(args.start, args.end + 1):
            nfl = "" if n_flow == 0 else f"_nFl{n_flow}"
            file = data_folder / f"{prob}_{algo}_N{n_particles}_pyMC_run{run_no}{nfl}_NS{N_SAMPLES}_theta_Exp{exp}.npy"

            if not file.exists():
                continue

            theta = np.load(str(file))
            if theta.ndim == 1:
                theta = theta[None, :]

            k = theta[:, 15]
            ll = (k == 1).cumsum() / np.arange(1, k.shape[0] + 1)

            m1problist.setdefault(f"{algo}_nFL{n_flow}", []).append(ll)

    return m1problist


def plot_rjmcmc_probabilities_estimates(
    data_folder,
    figures_folder,
    prob: str,
    algorithms: list[str],
    algorithms_colors: list[str],
    algorithms_labels: list[str],
    n_flows: list[int] | None,
    n_particles: int,
    exps: list[int],
    figsize: tuple[int, int] = (5, 3),
    suffix: str = "",
) -> None:
    plt.close()
    plt.style.use("seaborn-v0_8-white")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    for exp in exps:
        m1problist = generate_m1problist(data_folder, prob, algorithms, n_flows, n_particles, exp)

        for algo in m1problist:
            stack = np.column_stack(m1problist[algo]).T
            steps = np.arange(stack.shape[1])
            if False is True:
                mean = stack.mean(axis=0)
                std = stack.std(axis=0)
                ax.plot(steps, mean, color=algorithms_colors[algo], linewidth=0.5, label=algo)
                ax.fill_between(steps, mean - std, mean + std, alpha=0.3, facecolor=algorithms_colors[algo])
            else:
                ax.plot(
                    steps, stack.T, color=algorithms_colors[algo], linewidth=1, alpha=0.5, label=algorithms_labels[algo]
                )

    ax.hlines(
        0.882,
        xmin=-5000,
        xmax=5000 + N_SAMPLES,
        color="black",
        linewidth=1,
        linestyle="solid",
        label="Ground Truth",
    )

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=True))
    plt.legend(by_label.values(), by_label.keys(), loc="lower right", fancybox=True, frameon=True, framealpha=0.7)
    plt.xlabel("RJMCMC Step")
    plt.ylabel("2-Factor Model Probability Estimate")

    # Save figure
    plt.tight_layout()
    exp = f"Exp{exps[0]}" if len(exps) == 1 else f"Exps{'-'.join([str(exp) for exp in exps])}"
    fig.savefig(str(figures_folder / f"{prob}_rjmcmc_N{n_particles}_{exp}{suffix}.pdf"))
    plt.show()


if __name__ == "__main__":
    args = setup_argparse()

    data_folder = Path("data") / "raw"
    figures_folder = Path("docs") / "figures"
    figures_folder.mkdir(parents=True, exist_ok=True)

    PROBLEMS = ["FA"]
    ALGORITHMS = [f"FactorAnalysisModel{algo}" for algo in ["LW", "AF", "NF", "VINF"]]

    for n_particles in NPARTICLES:
        for prob in PROBLEMS:
            for exp in range(args.start, args.end + 1):
                plot_rjmcmc_probabilities_estimates(
                    data_folder,
                    figures_folder,
                    prob,
                    ALGORITHMS,
                    ALGORITHM_COLORS,
                    ALGORITHM_LABELS,
                    None,
                    n_particles,
                    exps=[exp],
                    figsize=(5, 3),
                )

            # Multiple chains
            exps = []
            if exps:
                plot_rjmcmc_probabilities_estimates(
                    data_folder,
                    figures_folder,
                    prob,
                    ALGORITHMS,
                    ALGORITHM_COLORS,
                    ALGORITHM_LABELS,
                    None,
                    n_particles,
                    exps=exps,
                    figsize=(5, 3),
                )
