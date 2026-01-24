from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from scripts.vs.vs_base import BLOCK_SIZE, GOLD_LOG_PROBDICT, K_LABELS, NPARTICLES, setup_argparse


def plot_violin(dcts, goldlogprobdict, labels, nparticles, klabel, figsize, file_name):
    plt.style.use("seaborn-v0_8-white")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    fig, ax = plt.subplots(nrows=len(nparticles), ncols=len(klabel), figsize=figsize)
    for j, n_particles in enumerate(nparticles):
        for i, glp in goldlogprobdict.items():
            ax[j, i].hlines(np.exp(glp).mean(), xmin=-0.5, xmax=5.5, linewidth=0.5, color="black", label="Ground Truth")

            for idx, dct in enumerate(dcts):
                if not dct:
                    continue

                ax[j, i].violinplot(dct[n_particles][:, i], positions=[idx], vert=1, showmeans=True)

            ax[j, i].set_xticks(np.arange(0, len(labels)))
            if j == 3:
                ax[j, i].set_xticklabels(labels, rotation=90, fontsize=8)
            else:
                ax[j, i].set_xticklabels([])

            ax[j, i].set_ylim([0, 1])
            if i > 0:
                ax[j, i].set_yticklabels([])
            if j == 0:
                ax[j, i].set_title(rf"$k=({klabel[i]})$")

        ax[j, 0].set_ylabel(f"N={int(n_particles / 2)}\n\nEstimated Model Probability")

    # Create output directory using pathlib
    figures_folder = Path("docs") / "figures"
    figures_folder.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

    plt.tight_layout()
    fig.savefig(str(figures_folder / file_name))
    # plt.show()


def prepare_dcts(folder, problems, algorithms, nparticles, block_size, exp):
    args = setup_argparse()

    dcts = [{} for _ in range(len(algorithms))]

    for n_particles in nparticles:
        for dct, prob, algo in zip(dcts, problems, algorithms, strict=True):
            lp_ss_s = []
            for run_no in range(args.start, args.end):
                file = folder / f"{prob}_{algo}_N{n_particles}_run{run_no}_BS{block_size}_Exp{exp}.txt"
                if not file.exists():
                    continue

                lp_ss = np.loadtxt(str(file), delimiter=";", usecols=[2, 4, 6, 8])
                lp_ss_s.append(lp_ss)

            if lp_ss_s:
                dct[n_particles] = np.exp(np.array(lp_ss_s))
    return dcts


if __name__ == "__main__":
    folder = Path("data") / "raw"
    folder.mkdir(exist_ok=True)

    PROBLEMS = ["VS", "VS", "VS", "VS", "VS", "VSC"]
    ALGORITHMS = [f"RobustBlockVSModel{algo}" for algo in ["Naive", "IndivAF", "IndivRQ", "CNF", "IndivVINF", "VINF"]]
    ALGORITHM_LABELS = ["Standard", "A-TRJ", "RQMA-TRJ", "RQMA-CTRJ", "Vinf with NFs", "Vinf with CNFs"]

    args = setup_argparse()
    for exp in range(args.start, args.end + 1):
        dcts = prepare_dcts(folder, PROBLEMS, ALGORITHMS, NPARTICLES, BLOCK_SIZE, exp)

        figsize = (8, 8.5)
        file_name = f"VS_VSC_variability_violin_Exp{exp}.pdf"
        plot_violin(dcts, GOLD_LOG_PROBDICT, ALGORITHM_LABELS, NPARTICLES, K_LABELS, figsize, file_name)
