from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from scripts.fa.fa_base import ALGORITHM_LABELS, BLOCK_SIZE, NPARTICLES, setup_argparse


def generate_m1probs_m2probs(
    data_folder, prob: str, algorithms: list[str], nsamp: list[int], n_flows: list[int], exp: int
):
    args = setup_argparse()

    if n_flows is None:
        n_flows = [0] * len(algorithms)

    m1probs, m2probs = {}, {}

    for algo, n_flow in zip(algorithms, n_flows, strict=True):
        for n_particles in nsamp:
            for run_no in range(args.start, args.end + 1):
                nfl = "" if n_flow == 0 else f"_nFl{n_flow}"
                file = f"{prob}_{algo}_N{n_particles}_pyMC_NUTS_run{run_no}{nfl}_BS{BLOCK_SIZE}_Exp{exp}.txt"
                file = data_folder / file

                if not file.exists():
                    return {}, {}

                m = np.loadtxt(str(file), usecols=[2, 4], delimiter=";")
                if m.ndim == 1:
                    m = m[None, :]

                m1probs.setdefault(algo, {}).setdefault(n_particles, []).append(np.exp(m[:, 0]))
                m2probs.setdefault(algo, {}).setdefault(n_particles, []).append(np.exp(m[:, 1]))

    return m1probs, m2probs


def plot_rjbridge_probability_estimates(
    data_folder,
    figures_folder,
    prob: str,
    algorithms: list[str],
    algorithms_labels: list[str],
    n_flows: list[int],
    n_particles: int,
    exp: int,
    figsize: tuple[int, int] = (6, 2.5),
    suffix: str = "",
) -> None:
    plt.close()
    plt.style.use("seaborn-v0_8-white")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    nsamp = [n_particles, NPARTICLES[-1]]
    fig, ax = plt.subplots(nrows=1, ncols=len(nsamp), figsize=figsize, sharey=True)
    ax = ax.flatten()

    m1probs = generate_m1probs_m2probs(data_folder, prob, algorithms, nsamp, n_flows, exp)[0]
    if not m1probs:
        return None

    algorithms = list(m1probs.keys())
    for idx_algo, algo in enumerate(algorithms):
        for idx, n_particles in enumerate(nsamp):
            m = np.hstack(m1probs[algo][n_particles])

            ax[idx].violinplot(m, positions=[idx_algo], showmeans=True)

            ax[idx].set_xticks(np.arange(len(algorithms)))
            ax[idx].set_xticklabels([algorithms_labels[algo] for algo in algorithms], rotation=45)
            ax[idx].set_title(f"N={n_particles} Per Model")

            ax[idx].set_ylim([0.5, 1.01])
            h = ax[idx].hlines(0.88, -0.5, len(algorithms) - 0.5, linewidth=0.5, color="black", label="Ground Truth")
            if idx == 1:
                ax[idx].legend(loc="lower right", handles=[h], fancybox=True, frameon=True, framealpha=0.7)

    ax[0].set_ylabel("2-Factor Model\nProbability Estimate")

    plt.tight_layout()
    fig.savefig(str(figures_folder / f"{prob}_rjbridge_N{'-'.join([str(n) for n in nsamp])}_Exp{exp}{suffix}.pdf"))
    # plt.show()

    return None


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
                plot_rjbridge_probability_estimates(
                    data_folder,
                    figures_folder,
                    prob,
                    ALGORITHMS,
                    ALGORITHM_LABELS,
                    None,
                    n_particles,
                    exp,
                    figsize=(6, 2.5),
                )
