from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from scripts.vs.plot_bbe_violin_all import prepare_dcts
from scripts.vs.vs_base import BLOCK_SIZE, GOLD_LOG_PROBDICT, K_LABELS, NPARTICLES, setup_argparse


def plot_violin_some(dcts, goldlogprobdict, labels, nparticles, klabel, figsize, file_name):
    plt.style.use("seaborn-v0_8-white")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    fig, ax = plt.subplots(nrows=len(klabel), ncols=len(nparticles), figsize=figsize)
    for j, n_particles in enumerate(nparticles):
        for i, glp in goldlogprobdict.items():

            ax[i, j].hlines(np.exp(glp).mean(), xmin=-0.5, xmax=3.5, linewidth=0.5, color="black", label="Ground Truth")

            for idx, dct in enumerate(dcts):
                if not dct:
                    continue

                ax[i, j].violinplot(dct[n_particles][:, i], positions=[idx], vert=1, showmeans=True)

            ax[i, j].set_xticks(np.arange(0, len(labels)))
            ax[i, j].set_xticklabels(labels, rotation=45, fontsize=8)

            ax[i, j].set_ylim([0, 1])
            if j > 0:
                ax[i, j].set_yticklabels([])

            ax[i, j].set_title(rf"$k=({klabel[i]})$")

        if j == 0:
            ax[0, 0].set_ylabel(f"N={int(n_particles / 2)}\n\nEstimated Model Probability")

    # Create output directory using pathlib
    figures_folder = Path("docs") / "figures"
    figures_folder.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

    plt.tight_layout()
    fig.savefig(str(figures_folder / file_name))
    # plt.show()


if __name__ == "__main__":
    folder = Path("data") / "raw"
    folder.mkdir(exist_ok=True)

    PROBLEMS = ["VS", "VS", "VS", "VS", "VS", "VSC"]
    ALGORITHMS = [f"RobustBlockVSModel{algo}" for algo in ["Naive", "IndivAF", "IndivRQ", "CNF", "IndivVINF", "VINF"]]
    ALGORITHM_LABELS = ["Standard", "A-TRJ", "RQMA-TRJ", "RQMA-CTRJ", "Vinf with NFs", "Vinf with CNFs"]

    nparticles = [NPARTICLES[0], NPARTICLES[-1]]  # [1000, 8000]
    idx = 3
    k_labels = [K_LABELS[3]]  # ["1,1,1,1"]
    gold_log_probdict = {idx: GOLD_LOG_PROBDICT[idx]}

    args = setup_argparse()
    for exp in range(args.start, args.end + 1):
        dcts = prepare_dcts(folder, PROBLEMS, ALGORITHMS, nparticles, BLOCK_SIZE, exp)

        figsize = (6, 3.5)
        file_name = f"VS_VSC_variability_violin_Exp{exp}_{int(nparticles[0]/2)}_{int(nparticles[-1]/2)}.pdf"
        plot_violin_some(dcts, gold_log_probdict, ALGORITHM_LABELS, nparticles, k_labels, figsize, file_name)
