from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from scripts.sas.sas_base import DEFAULT_DATA_DICT, configure_flow_training, setup_argparse
from scripts.sas.sas_vinfs import get_normalizing_flows
from src.tools import PlotSinharcsinhPropAll
from src.utils.tools import kde_1D, kde_joint

N_SAMPLES = 100000


def plot_proposal(prob, prob_dict, exp, n_samples, figsize=(6, 6)):
    # Set font types for publication quality figures
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    plt.close()
    nd = 30

    # Generate data
    _plot = PlotSinharcsinhPropAll(prob, seed=exp, nd=nd, n_samples=n_samples)
    m1theta, m2theta, p_sas_1d, p_u, pt_prop_m1theta = _plot.run(
        list(prob_dict.keys()), verbose=False, normalizing_flows=get_normalizing_flows
    )

    # Create visualization
    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=figsize)
    ax = ax.flatten()

    # Plot 1D source distribution
    ax[0].set_title("Source 1D SAS")
    kde_1D(ax[0], m1theta[:, 0], bw=1, plotline=False)
    ax[0].set_yticklabels([])

    # Scatter plot of grid points colored by probability
    ax[0].scatter(p_sas_1d, np.zeros_like(p_u), c=p_u, marker="x", cmap="brg")

    # Plot 2D proposal distributions for each proposal type
    for i, algo in enumerate(prob_dict):
        # Plot KDE of true model 2 distribution
        kde_joint(ax[i + 1], m2theta[:, [0, 2]], cmap="Blues", alpha=1, bw=0.1, maxz_scale=2, n_grid_points=128)

        # Scatter plot of proposals colored by source probability
        ax[i + 1].scatter(
            pt_prop_m1theta[algo][:, 0],
            pt_prop_m1theta[algo][:, 2],
            s=0.3,
            alpha=0.5,
            c=np.tile(p_u, nd),
            cmap="brg",
        )

        # Set axis limits for consistent visualization
        ax[i + 1].set_xlim([-1, 15])
        ax[i + 1].set_ylim([-7, 0.5])
        ax[i + 1].set_title(prob_dict[algo]["title"])

    # Create output directory using pathlib
    figures_folder = Path("docs") / "figures"
    figures_folder.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

    # Adjust layout and save figure
    plt.tight_layout()
    fig.savefig(str(figures_folder / f"SAS_proposal_NS{n_samples}_Exp{exp}.pdf"))
    # plt.show()


if __name__ == "__main__":
    args = setup_argparse()
    configure_flow_training(
        args.flow_device,
        flow_num_samples=args.flow_num_samples,
        flow_hidden_layer_size=args.flow_hidden_layer_size,
        flow_annealing=args.flow_annealing,
    )
    for prob, prob_dict in DEFAULT_DATA_DICT.items():
        for exp in range(args.start, args.end + 1):
            plot_proposal(prob, prob_dict, exp, N_SAMPLES, figsize=(6, 6))
