import argparse

import numpy as np
import pymc as pm
from pytensor import tensor as at

from src.main import Experiments

NPARTICLES = [2000, 4000, 8000, 16000]
N_SAMPLES = 100000
BLOCK_SIZE = 1000

ALGORITHM_COLORS = {
    "FactorAnalysisModelVINF": "blue",
    "FactorAnalysisModelLW": "pink",
    "FactorAnalysisModelAF": "darkorange",
    "FactorAnalysisModelNF": "green",
}
ALGORITHM_LABELS = {
    "FactorAnalysisModelVINF": "VI with nflows",
    "FactorAnalysisModelLW": "Lopes & West",
    "FactorAnalysisModelAF": "Affine TRJ",
    "FactorAnalysisModelNF": "RQMA-NF TRJ",
}


def setup_argparse():
    parser = argparse.ArgumentParser(description="Process start and end parameters")
    parser.add_argument("--start", type=int, nargs="?", default=1, help="Start value (default: 1)")
    parser.add_argument("--end", type=int, nargs="?", default=3, help="End value (default: 3)")
    return parser.parse_args()


def generate_train_theta_test_theta_k1(folder, run_no, n_particles, prob, y_data, SMC_SAMPLE_KWARGS):
    # generate SMC2 posterior for 2-factor model
    draws = SMC_SAMPLE_KWARGS["draws"]
    chains = SMC_SAMPLE_KWARGS["chains"]
    k = 1

    fa2f_model = pm.Model()
    with fa2f_model:
        betaii = pm.HalfNormal("betaii", sigma=1.0, shape=2)
        betaij = pm.Normal("betaij", mu=0.0, sigma=1.0, shape=5 + 4)
        lambdaii = pm.InverseGamma("lambdaii", 1.1, 0.05, shape=(6))
        mu = at.zeros(6)
        beta1 = at.stack([betaii[0]] + [betaij[i] for i in range(5)], axis=0)
        beta2 = at.stack([0, betaii[1]] + [betaij[i] for i in range(5, 9)], axis=0)
        beta = at.stack([beta1, beta2], axis=1)
        sigma = at.diag(lambdaii) + at.dot(beta, beta.T)
        obs = pm.MvNormal("obs", mu=mu, cov=sigma, observed=y_data)

    with fa2f_model:
        fa2f_trace = pm.sample_smc(**SMC_SAMPLE_KWARGS)
        # arrange parameters in format for RJBridge
        p = fa2f_trace["posterior"]
        betaii = p["betaii"]
        betaij = p["betaij"]
        lambdaii = p["lambdaii"]
        for gid, group in enumerate(["test", "train"]):
            thlist = []
            for i in range(int(gid * chains / 2), int((gid + 1) * chains / 2)):
                thlist.append(
                    np.column_stack(
                        [
                            betaii[i, :, 0],
                            betaij[i, :, 0:5],
                            betaii[i, :, 1],
                            betaij[i, :, 5:9],
                            np.zeros((draws, 4)),
                            np.ones(draws) * k,
                            lambdaii[i],
                        ]
                    )
                )
            m2theta = np.vstack(thlist)
            file = str(folder / f"{prob}_pyMC_k{k}_N{n_particles}_run{run_no}_{group}.npy")
            np.save(file, m2theta)

    return None


def generate_train_theta_test_theta_k2(folder, run_no, n_particles, prob, y_data, NUTS_SAMPLE_KWARGS):
    # generate NUTS posterior for 3-factor model
    draws = NUTS_SAMPLE_KWARGS["draws"]
    chains = NUTS_SAMPLE_KWARGS["chains"]
    k = 2
    fa3f_model = pm.Model()

    with fa3f_model:
        betaii = pm.HalfNormal("betaii", sigma=1.0, shape=3)
        betaij = pm.Normal("betaij", mu=0.0, sigma=1.0, shape=5 + 4 + 3)
        lambdaii = pm.InverseGamma("lambdaii", 1.1, 0.05, shape=(6))
        mu = at.zeros(6)
        beta1 = at.stack([betaii[0]] + [betaij[i] for i in range(5)], axis=0)
        beta2 = at.stack([0, betaii[1]] + [betaij[i] for i in range(5, 9)], axis=0)
        beta3 = at.stack([0, 0, betaii[2]] + [betaij[i] for i in range(9, 12)], axis=0)
        beta = at.stack([beta1, beta2, beta3], axis=1)
        sigma = at.diag(lambdaii) + at.dot(beta, beta.T)
        obs = pm.MvNormal("obs", mu=mu, cov=sigma, observed=y_data)

    with fa3f_model:
        fa3f_trace = pm.sample(**NUTS_SAMPLE_KWARGS)

        # arrange parameters in format for RJBridge
        p = fa3f_trace["posterior"]
        betaii = p["betaii"]
        betaij = p["betaij"]
        lambdaii = p["lambdaii"]
        for gid, group in enumerate(["test", "train"]):
            thlist = []
            for i in range(int(gid * chains / 2), int((gid + 1) * chains / 2)):
                thlist.append(
                    np.column_stack(
                        [
                            betaii[i, :, 0],
                            betaij[i, :, 0:5],
                            betaii[i, :, 1],
                            betaij[i, :, 5:9],
                            betaii[i, :, 2],
                            betaij[i, :, 9:],
                            np.ones(draws) * k,
                            lambdaii[i],
                        ]
                    )
                )
            m3theta = np.vstack(thlist)
            file = str(folder / f"{prob}_pyMC_k{k}_N{n_particles}_run{run_no}_{group}.npy")
            np.save(file, m3theta)

    return None


def get_train_theta_test_theta(folder, run_no, prob, n_particles):
    mk_theta_train = {}
    mk_theta_test = {}
    for k in range(1, 3):  # k=1 and k=2
        mk_theta_train[k] = np.load(str(folder / f"{prob}_pyMC_k{k}_N{n_particles}_run{run_no}_train.npy"))
        mk_theta_test[k] = np.load(str(folder / f"{prob}_pyMC_k{k}_N{n_particles}_run{run_no}_test.npy"))
    train_theta = np.vstack([th for mk, th in mk_theta_train.items()])
    test_theta = np.vstack([th for mk, th in mk_theta_test.items()])
    return train_theta, test_theta


def get_train_theta_start_theta(folder, run_no, prob, n_particles):
    train_theta, test_theta = get_train_theta_test_theta(folder, run_no, prob, n_particles)

    start_theta = test_theta[np.random.choice(test_theta.shape[0], size=1)]
    return train_theta, start_theta


def run_rjmcmc_algorithm_based_ablation(
    task_config, *, problems, algorithms, start, end, ablations, train_theta, start_theta, n_samples, y_data, **kwargs
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
        y_data=y_data,
        normalizing_flows=ablations[ablation_id]["normalizing_flows"],
        **kwargs,
    )


def run_rjbridge_algorithm_based_ablation(
    task_config, *, problems, algorithms, start, end, ablations, train_theta, test_theta, block_size, y_data, **kwargs
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
        y_data=y_data,
        normalizing_flows=ablations[ablation_id]["normalizing_flows"],
        **kwargs,
    )
