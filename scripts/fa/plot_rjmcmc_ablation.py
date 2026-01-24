from pathlib import Path

from matplotlib import colors

from scripts.fa.fa_base import NPARTICLES, setup_argparse
from scripts.fa.plot_fa_rjmcmc import plot_rjmcmc_probabilities_estimates

if __name__ == "__main__":
    args = setup_argparse()

    data_folder = Path("data") / "raw"
    figures_folder = Path("docs") / "figures"
    figures_folder.mkdir(parents=True, exist_ok=True)

    N_FLOWS = list(range(2, 45, 2))

    PROBLEMS = ["FA"]
    ALGORITHMS = [f"FactorAnalysisModel{algo}" for algo in ["VINF"]] * len(N_FLOWS)
    colors_matplotlib = [name for name, _ in colors.cnames.items()]
    ALGORITHM_COLORS = {f"{ALGORITHMS[0]}_nFL{n}": colors_matplotlib[idx] for idx, n in enumerate(N_FLOWS)}
    ALGORITHM_LABELS = {f"{ALGORITHMS[0]}_nFL{n}": f"VI with nflows {n}" for n in N_FLOWS}

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
                    N_FLOWS,
                    n_particles,
                    exps=[exp],
                    figsize=(10, 6),
                    suffix="_ablation",
                )
