from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from scripts.sas.sas_base import DEFAULT_DATA_DICT, N_SAMPLES, setup_argparse


def plot_running_mp_trace(prob, prob_dict, exps, steps, n_samples, figsize=(5, 3), suffix=""):
    # Set font types for publication quality figures
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    plt.close()

    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    folder = Path("data") / "raw"

    for exp in exps:
        for algo, algo_dict in prob_dict.items():
            file_path = folder / f"{prob}_{algo}_NS{n_samples}_theta_Exp{exp}.npy"
            pt_m1theta = np.load(str(file_path))
            trace = pt_m1theta[:, 1].cumsum() / steps
            ax.plot(
                steps,
                trace,
                color=algo_dict["color"],
                linewidth=0.5,
                label=algo_dict["title"],
                alpha=algo_dict["alpha"],
            )

    # plt.suptitle('Sinh Arcsinh 1D 2D Jump 1->2 PLMA Transform')

    # Add horizontal line for ground truth
    ax.hlines(
        0.75,
        xmin=-500,
        xmax=n_samples + 500,
        color="black",
        linewidth=1,
        linestyle="solid",
        label="Ground Truth",
    )

    # Create legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=True))
    plt.legend(by_label.values(), by_label.keys(), loc="lower right", fancybox=True, frameon=True, framealpha=0.7)

    # Set plot labels and limits
    ax.set_ylabel(r"$k=2$ SAS Model Probability Estimate")
    ax.set_xlabel("RJMCMC Step")
    ax.set_ylim((0.55, 0.85))

    # Create output directory using pathlib
    figures_folder = Path("docs") / "figures"
    figures_folder.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

    # Save figure
    plt.tight_layout()
    exp = f"Exp{exps[0]}" if len(exps) == 1 else f"Exps{'-'.join([str(exp) for exp in exps])}"
    fig.savefig(str(figures_folder / f"{prob}_running_mp_trace_NS{n_samples}_{exp}{suffix}.pdf"))
    # plt.show()


if __name__ == "__main__":
    steps = np.arange(1, N_SAMPLES + 1)

    args = setup_argparse()
    for prob, prob_dict in DEFAULT_DATA_DICT.items():
        for exp in range(args.start, args.end + 1):
            plot_running_mp_trace(prob, prob_dict, [exp], steps, N_SAMPLES, figsize=(5, 3), suffix="")

        # Multiple chains
        exps = []
        if exps:
            plot_running_mp_trace(prob, prob_dict, exps, steps, N_SAMPLES, figsize=(5, 3), suffix="")
