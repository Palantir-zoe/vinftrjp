import argparse
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from src.algorithms import get_algorithm
from src.main import Experiments
from src.problems import get_problem
from src.samplers import SMC1

NPARTICLES = [1000, 2000, 4000, 8000]
K_LIST = [[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
K_LABELS = [f"{k[0]},{k[1]},{k[2]},{k[2]}" for k in K_LIST]
BLOCK_SIZE = 1000
BASE_ALGORITHM = "RobustBlockVSModelIndivSMC"

# GOLD_LOG_PROBDICT
file = str(Path("data") / "core" / "SMCLogZ_BlockVS_indiv.txt")
smc_logz = np.loadtxt(file, delimiter=";", usecols=[0, 3])
p2smc_logz = [0, 2, 3, 6]  # 0:[1, 0, 0, 0]; 2:[1, 0, 1, 0]; 3:[1, 1, 0, 0]; 6:[1, 1, 1, 1]


def prepare_goldlogprobdict(smc_logz, p2smc_logz):
    nmodels = len(p2smc_logz)  # p2smc_logz = [0, 2, 3, 6]

    lzdict = {}
    for pi, lzi in enumerate(p2smc_logz):  # p2smc_logz = [0, 2, 3, 6]
        lzdict[pi] = smc_logz[smc_logz[:, 0] == lzi, 1]

    lzmeans = np.zeros(nmodels)
    for i, lz in lzdict.items():
        lzmeans[i] = logsumexp(lz) - np.log(lz.shape[0])

    goldlogprobdict = {}
    for i, lz in lzdict.items():
        goldlogprobdict[i] = lz - logsumexp(lzmeans)

    return goldlogprobdict


goldlogprobdict = prepare_goldlogprobdict(smc_logz, p2smc_logz)
GOLD_LOG_PROBDICT = goldlogprobdict


def setup_argparse():
    parser = argparse.ArgumentParser(description="Process start and end parameters")
    parser.add_argument("--start", type=int, nargs="?", default=1, help="Start value (default: 1)")
    parser.add_argument("--end", type=int, nargs="?", default=3, help="End value (default: 3)")
    return parser.parse_args()


def generate_train_theta_test_theta(folder, run_no, n_particles, prob, k_list, base_algorithm):
    for k in k_list:
        problem = get_problem(prob)
        model = get_algorithm(base_algorithm, problem=problem, k=k)
        smc = SMC1(model)

        name = f"{prob}_{base_algorithm}_k{k[0]}{k[1]}{k[2]}_N{n_particles}_run{run_no}.npy"
        final_theta = smc.run(n_particles, 0.5)[0]
        np.save(str(folder / name), final_theta)

    return None


def get_train_theta_test_theta(folder, run_no, n_particles, prob, k_list, base_algorithm):
    theta_list = []
    for k in k_list:
        name = f"{prob}_{base_algorithm}_k{k[0]}{k[1]}{k[2]}_N{n_particles}_run{run_no}.npy"
        theta_list.append(np.load(str(folder / name)))

    theta = np.vstack(theta_list)
    train_idx = np.random.choice(theta.shape[0], size=int(theta.shape[0] / 2), replace=False)
    test_idx = test_idx = np.delete(np.arange(theta.shape[0]), train_idx)
    train_theta = theta[train_idx]
    test_theta = theta[test_idx]

    return train_theta, test_theta


def run_rjmcmc_algorithm_based_ablation(
    task_config, *, problems, algorithms, start, end, ablations, train_theta, start_theta, n_samples, **kwargs
) -> None:
    if len(algorithms) != len(ablations):
        raise ValueError(f"The lenght of {algorithms} and {ablations} should be equal.")

    problem_id = task_config.problem_id  # None
    algorithm_id = task_config.algorithm_id  # is not None

    p_start, p_end = 0, len(problems)
    if problem_id is not None:
        p_start, p_end = problem_id, problem_id + 1
    _problems = problems[p_start:p_end]

    a_start, a_end = 0, len(algorithms)
    if algorithm_id is not None:
        a_start, a_end = algorithm_id, algorithm_id + 1
    _algorithms = algorithms[a_start:a_end]

    ablation_id: int = problem_id if problem_id is not None else algorithm_id

    config = {"start": start, "end": end}

    # Problem-specific experiment execution
    e = Experiments(problems=_problems, algorithms=_algorithms, **config)
    e.run(
        algorithm_suffix=ablations[ablation_id]["algorithm_suffix"],
        # rjmcmc
        calibrate_draws=train_theta,
        start_theta=start_theta,
        n_samples=n_samples,
        # algorithm
        normalizing_flows=ablations[ablation_id]["normalizing_flows"],
        **kwargs,
    )


def run_rjbridge_algorithm_based_ablation(
    task_config, *, problems, algorithms, start, end, ablations, train_theta, test_theta, block_size, **kwargs
) -> None:
    if len(algorithms) != len(ablations):
        raise ValueError(f"The lenght of {algorithms} and {ablations} should be equal.")

    problem_id = task_config.problem_id  # None
    algorithm_id = task_config.algorithm_id  # is not None

    p_start, p_end = 0, len(problems)
    if problem_id is not None:
        p_start, p_end = problem_id, problem_id + 1
    _problems = problems[p_start:p_end]

    a_start, a_end = 0, len(algorithms)
    if algorithm_id is not None:
        a_start, a_end = algorithm_id, algorithm_id + 1
    _algorithms = algorithms[a_start:a_end]

    ablation_id: int = problem_id if problem_id is not None else algorithm_id

    config = {"start": start, "end": end}

    # Problem-specific experiment execution
    e = Experiments(problems=_problems, algorithms=_algorithms, **config)
    e.run(
        algorithm_suffix=ablations[ablation_id]["algorithm_suffix"],
        # rjbridge
        train_theta=train_theta,
        test_theta=test_theta,
        block_size=block_size,
        # algorithm
        normalizing_flows=ablations[ablation_id]["normalizing_flows"],
        **kwargs,
    )
