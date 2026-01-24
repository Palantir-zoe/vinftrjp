from pathlib import Path

import numpy as np

from scripts.vs.plot_bbe_violin_all import plot_violin
from scripts.vs.vs_base import BLOCK_SIZE, GOLD_LOG_PROBDICT, K_LABELS, NPARTICLES, setup_argparse


def prepare_dcts(folder, n_flows, problems, algorithms, nparticles, block_size, exp):
    args = setup_argparse()

    dcts = [{} for _ in range(len(algorithms))]

    for n_particles in nparticles:
        for dct, n_flow, prob, algo in zip(dcts, n_flows, problems, algorithms, strict=True):
            lp_ss_s = []
            for run_no in range(args.start, args.end):
                file = str(folder / f"{prob}_{algo}_N{n_particles}_run{run_no}_nFL{n_flow}_BS{block_size}_Exp{exp}.txt")
                lp_ss = np.loadtxt(file, delimiter=";", usecols=[2, 4, 6, 8])
                lp_ss_s.append(lp_ss)
            dct[n_particles] = np.exp(np.array(lp_ss_s))
    return dcts


if __name__ == "__main__":
    folder = Path("data") / "raw"
    folder.mkdir(exist_ok=True)

    N_FLOWS = list(range(2, 45, 2))
    PROBLEMS = ["VS"] * len(N_FLOWS)
    ALGORITHMS = [f"RobustBlockVSModel{algo}" for algo in ["IndivVINF"]] * len(N_FLOWS)
    ALGORITHM_LABELS = [f"Vinf with NFs {n}" for n in N_FLOWS]

    args = setup_argparse()
    for exp in range(args.start, args.end + 1):
        dcts = prepare_dcts(folder, N_FLOWS, PROBLEMS, ALGORITHMS, NPARTICLES, BLOCK_SIZE, exp)

        figsize = (18, 18.5)
        file_name = f"VS_variability_violin_Exp{exp}_ablation.pdf"
        plot_violin(dcts, GOLD_LOG_PROBDICT, ALGORITHM_LABELS, NPARTICLES, K_LABELS, figsize, file_name)
