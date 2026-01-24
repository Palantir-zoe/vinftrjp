from pathlib import Path

from scripts.fa.fa_base import NPARTICLES, setup_argparse
from scripts.fa.plot_fa_rjbridge import plot_rjbridge_probability_estimates

if __name__ == "__main__":
    args = setup_argparse()

    data_folder = Path("data") / "raw"
    figures_folder = Path("docs") / "figures"
    figures_folder.mkdir(parents=True, exist_ok=True)

    N_FLOWS = list(range(2, 45, 2))

    PROBLEMS = ["FA"]
    ALGORITHMS = [f"FactorAnalysisModel{algo}" for algo in ["VINF"]] * len(N_FLOWS)
    ALGORITHM_LABELS = {f"{ALGORITHMS[0]}_nFL{n}": "VI with nflows {n}" for n in N_FLOWS}

    for n_particles in NPARTICLES:
        for prob in PROBLEMS:
            for exp in range(args.start, args.end + 1):
                plot_rjbridge_probability_estimates(
                    data_folder,
                    figures_folder,
                    prob,
                    ALGORITHMS,
                    ALGORITHM_LABELS,
                    N_FLOWS,
                    n_particles,
                    exp,
                    figsize=(10, 6),
                    suffix="_ablation",
                )
