import time

import numpy as np

from .experiment import Experiment


class Experiments:
    def __init__(self, start: int = 1, end: int = 20, problems=None, algorithms=None, device="cpu"):
        MAXIMUM_TIMES = 1000  # maximum independent experiments times

        self.start = start  # starting index of independent experiments
        if self.start <= 0:
            raise ValueError(f"start index must be positive, got {self.start}")
        self.end = end  # ending index of independent experiments
        if not (self.end > 0 and self.end >= self.start and self.end <= MAXIMUM_TIMES):
            e = "end index must be positive, >= start"
            raise ValueError(f"{e} ({self.start}), and <= MAXIMUM_TIMES ({MAXIMUM_TIMES}), got {self.end}")
        self.indices = range(self.start, self.end + 1)  # index range (1-based rather 0-based)

        if problems is None:
            problems = ["SAS"]
        self.problems = problems

        if algorithms is None:
            algorithms = ["ToyModelAF", "ToyModelNF", "ToyModelPerfect", "ToyModelVINF"]
        self.algorithms = algorithms

        self.device = device

        # seeds
        self.seeds = np.random.default_rng(2025).integers(  # to generate all random seeds in advances
            np.iinfo(np.int32).max, size=(len(self.problems), MAXIMUM_TIMES)
        )

    def run(self, **kwargs) -> None:
        for index in self.indices:
            print(f"* Experiment: {index:d} ***:")
            for algorithm in self.algorithms:
                print(f"    * Algorithm: {algorithm:s} ***:")
                for idx_prob, problem in enumerate(self.problems):
                    start_time = time.time()

                    print(f"      * Problem: {problem:s} ***:")
                    seed = int(self.seeds[idx_prob, index - 1])  # 1-based

                    e = Experiment(problem, index=index, seed=seed, device=self.device)
                    e.run(algorithm, **kwargs)

                    print(f"        runtime: {time.time() - start_time:7.5e}.\n")

        return None
