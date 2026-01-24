import random

import numpy as np
import torch
from scipy.stats import norm

from src.algorithms import get_algorithm
from src.problems import get_problem
from src.samplers import RJMCMC
from src.transforms import SinArcSinhTransform


class PlotSinharcsinhPropAll:
    def __init__(self, problem: str, seed: int = 2025, nd=30, n_samples=10000):
        self.problem = problem
        self.nd = nd
        self.n_samples = n_samples
        self.seed = seed

    def run(self, algorithms: list[str], **kwargs):
        self.set_random_seed(self.seed)

        calibrate_draws, m1theta, m2theta = self.generate_calibration_data(**kwargs)
        p_sas_1d, p_u, p_sas = self.generate_p_sas_1d()

        pt_prop_m1theta = self.generate_pt_prop_m1theta(p_sas, calibrate_draws, algorithms, **kwargs)
        return m1theta, m2theta, p_sas_1d, p_u, pt_prop_m1theta

    def generate_calibration_data(self, algo: str = "ToyModelAF", **kwargs):
        run_index = 0

        # Generate calibration data
        prob = get_problem(self.problem)  # Create problem instance
        model = get_algorithm(algo, problem=prob, run_index=run_index, **kwargs)  # Create algorithm instance
        calibrate_draws = model.draw_perfect(self.n_samples)

        # Separate samples by model index (m1 and m2)
        m1idx = calibrate_draws[:, 1] == 0  # Index for model 1 samples
        m2idx = calibrate_draws[:, 1] == 1  # Index for model 2 samples
        m1theta = calibrate_draws[m1idx]  # Parameters for model 1
        m2theta = calibrate_draws[m2idx]  # Parameters for model 2
        return calibrate_draws, m1theta, m2theta

    def generate_p_sas_1d(self):
        # Create SinArcSinh transformation for 1D visualization
        prob = get_problem(self.problem)  # Create algorithm instance
        sastf = SinArcSinhTransform(prob.ep[0], prob.dp[0])

        # Generate grid points for evaluation
        p_u = np.linspace(1e-5, 1 - 1e-5, self.nd)  # Uniform grid in probability space
        p_n = norm(0, 1).ppf(p_u)  # Transform to normal space
        p_sas_1d, _ = sastf.forward(torch.Tensor(p_n))  # Apply SinArcSinh transformation
        p_sas_1d = p_sas_1d.detach().numpy()  # Convert to numpy array

        # Create extended parameter array for proposal testing
        p_sas = np.column_stack([p_sas_1d, np.zeros((self.nd, 2))])

        return p_sas_1d, p_u, p_sas

    def generate_pt_prop_m1theta(self, p_sas, calibrate_draws, algorithms: list[str], **kwargs):
        # Dictionary to store proposal results for each proposal type
        pt_prop_m1theta = {}

        # Test proposals for each model type
        for algo in algorithms:
            run_index = 0
            prob = get_problem(self.problem)  # Create problem instance
            model = get_algorithm(algo, problem=prob, run_index=run_index, **kwargs)  # Create algorithm instance

            # Initialize RJMCMC sampler with calibration data
            rjmcmc = RJMCMC(model, calibrate_draws=calibrate_draws)

            # Collect proposals for each grid point
            ptl = []
            for i in range(self.nd):
                # Update run index for both proposal and model
                model.proposal.set_run_index(i)  # Set run index in proposal component
                model.run_index = i  # Set run index in model itself

                # Generate proposal using RJMCMC
                prop_m1theta, _, _ = rjmcmc.pmodel.propose(p_sas, self.nd)
                ptl.append(prop_m1theta)

            # Store results for this proposal type
            pt_prop_m1theta[algo] = np.vstack(ptl)

        return pt_prop_m1theta

    @staticmethod
    def set_random_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
