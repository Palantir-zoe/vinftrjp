from pathlib import Path

from scripts.vs.plot_bbe_violin_all import plot_violin
from scripts.vs.plot_bbe_violin_all_vinf_ablation import prepare_dcts
from scripts.vs.vs_base import BLOCK_SIZE, GOLD_LOG_PROBDICT, K_LABELS, NPARTICLES, setup_argparse

if __name__ == "__main__":
    folder = Path("data") / "raw"
    folder.mkdir(exist_ok=True)

    N_FLOWS = list(range(2, 45, 2))
    PROBLEMS = ["VSC"] * len(N_FLOWS)
    ALGORITHMS = [f"RobustBlockVSModel{algo}" for algo in ["VINF"]] * len(N_FLOWS)
    ALGORITHM_LABELS = [f"Vinf with CNFs {n}" for n in N_FLOWS]

    args = setup_argparse()
    for exp in range(args.start, args.end + 1):
        dcts = prepare_dcts(folder, N_FLOWS, PROBLEMS, ALGORITHMS, NPARTICLES, BLOCK_SIZE, exp)

        figsize = (18, 18.5)
        file_name = f"VSC_variability_violin_Exp{exp}_ablation.pdf"
        plot_violin(dcts, GOLD_LOG_PROBDICT, ALGORITHM_LABELS, NPARTICLES, K_LABELS, figsize, file_name)
