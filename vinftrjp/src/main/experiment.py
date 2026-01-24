import os
import random

import numpy as np
import torch

from src.algorithms import get_algorithm
from src.problems import get_problem
from src.samplers import RJMCMC, RJBridge


class Experiment:
    def __init__(self, problem: str, index: int, seed: int, device="cpu") -> None:
        self.problem = problem

        self.index = index
        self.seed = seed

        self._folder_raw = os.path.join("data", "raw")
        os.makedirs(self._folder_raw, exist_ok=True)

        self.device = device

    def run(
        self,
        algorithm: str,
        *,
        algorithm_suffix: str = "",  # rjmcmc and rjbridge
        # rjmcmc
        calibrate_draws=None,
        n_samples=10000,
        start_theta=None,
        # rjbridge
        train_theta=None,
        test_theta=None,
        block_size=1000,
        **kwargs,
    ):
        """
        Unified interface for running RJMCMC (Reversible Jump Markov Chain Monte Carlo)
        or RJBridge (Reversible Jump Bridge Sampling) algorithms.

        The method automatically selects the appropriate algorithm based on the provided
        parameters: RJBridge if both train_theta and test_theta are provided, otherwise RJMCMC.

        Parameters
        ----------
        algorithm : str
            Name of the algorithm to execute. Must correspond to a registered algorithm
            in the algorithm registry.

        algorithm_suffix : str, optional
            String suffix appended to output filenames to distinguish different runs
            or configurations. Default is empty string.

        calibrate_draws : np.ndarray or None, optional
            Pre-computed parameter samples used to calibrate the proposal distributions
            in RJMCMC. Shape should be (n_calibration_samples, n_parameters).
            If None, proposal distributions are initialized with default settings.

        n_samples : int, optional
            Total number of MCMC samples to generate in RJMCMC mode. Default is 10000.

        start_theta : ndarray or None, optional
            Initial parameter values for RJMCMC chain.
            Shape: (1, n_parameters)
            If None, initial values are drawn from prior.

        train_theta : ndarray or None, optional
            Training parameter samples. For RJMCMC, used as additional calibration data.
            For RJBridge, used as source distribution for importance sampling.
            Shape: (n_train_samples, n_parameters)

        train_theta : np.ndarray or None, optional
            Source distribution samples for RJBridge importance sampling. These represent
            the "easy" distribution to sample from. Shape should be (n_train_samples, n_parameters).
            Required for RJBridge mode.

        test_theta : np.ndarray or None, optional
            Target distribution samples for RJBridge importance sampling. These represent
            the "hard" distribution we want to estimate. Shape should be (n_test_samples, n_parameters).
            Required for RJBridge mode.

        block_size : int, optional
            Number of samples to process in each batch during RJBridge computation.
            Larger values use more memory but may improve performance. Default is 1000.

        **kwargs : dict
            Additional keyword arguments passed to the algorithm constructor.

        Returns
        -------
        None
            Results are saved to output files rather than returned.
        """
        self.set_random_seed(self.seed, deterministic=True)

        prob = get_problem(self.problem)  # instance
        model = get_algorithm(algorithm, problem=prob, **kwargs)  # instance

        if (train_theta is not None) and (test_theta is not None):
            self.run_rjbridge(
                algorithm,
                algorithm_suffix,
                model,
                train_theta=train_theta,
                test_theta=test_theta,
                block_size=block_size,
            )
        else:
            self.run_rjmcmc(
                algorithm,
                algorithm_suffix,
                model,
                calibrate_draws=calibrate_draws,
                start_theta=start_theta,
                n_samples=n_samples,
            )

    def run_rjmcmc(
        self,
        algorithm,
        algorithm_suffix,
        model,
        calibrate_draws=None,
        start_theta=None,
        n_samples=10000,
    ):
        # file name for each experiment
        _file_name = "{}_{}_{}NS{}_{}_Exp{}.npy"

        # Generate calibration data from perfect model
        if calibrate_draws is None:
            calibrate_draws = model.draw_perfect(n_samples)

        # Initialize RJMCMC sampler with calibration data
        rjmcmc = RJMCMC(model, calibrate_draws=calibrate_draws)

        # Run MCMC sampling
        final_theta, prop_theta, _, _, ar = rjmcmc.run(n_samples, start_theta)

        # Save results for posterior analysis
        file_name = _file_name.format(self.problem, algorithm, algorithm_suffix, n_samples, "theta", self.index)
        np.save(os.path.join(self._folder_raw, file_name), final_theta)

        file_name = _file_name.format(self.problem, algorithm, algorithm_suffix, n_samples, "ptheta", self.index)
        np.save(os.path.join(self._folder_raw, file_name), prop_theta)

        file_name = _file_name.format(self.problem, algorithm, algorithm_suffix, n_samples, "ar", self.index)
        np.save(os.path.join(self._folder_raw, file_name), ar)

    def run_rjbridge(
        self,
        algorithm,
        algorithm_suffix,
        model,
        train_theta,
        test_theta,
        block_size=1000,
    ):
        # file name for each experiment
        _file_name = "{}_{}_{}BS{}_Exp{}.npy"

        rjb = RJBridge(model, train_theta)
        log_p_mk_dict = rjb.estimate_log_p_mk(test_theta, block_size)

        data_list = ["BE_lp;"]
        for mk, log_p_mk in log_p_mk_dict.items():
            data_list.append(str(mk) + ";" + str(log_p_mk) + ";")
        data_list.append("0\n")

        file_name = _file_name.format(self.problem, algorithm, algorithm_suffix, block_size, self.index)
        with open(os.path.join(self._folder_raw, file_name.replace(".npy", ".txt")), "w") as f:
            f.write("".join(data_list))

    def set_random_seed(self, seed, deterministic=True):
        os.environ["PYTHONHASHSEED"] = str(seed)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if self.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            torch.set_default_device("cpu")
            torch.set_default_dtype(torch.float32)

        elif torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                if hasattr(torch, "use_deterministic_algorithms"):
                    torch.use_deterministic_algorithms(True)
