import numpy as np

from scripts.sas.plot_sas_mp_trace import plot_running_mp_trace
from scripts.sas.sas_base import N_SAMPLES, setup_argparse

DEFAULT_DATA_DICT = {
    "SAS": {
        "ToyModelVINF_6and5": {"title": "VI with NFs 6-5", "color": "#1f77b4", "alpha": 1.0},
        "ToyModelVINF_6and8": {"title": "VI with NFs 6-8", "color": "#ff7f0e", "alpha": 0.9},
        "ToyModelVINF_6and11": {"title": "VI with NFs 6-11", "color": "#7f7f7f", "alpha": 0.8},
        "ToyModelVINF_9and5": {"title": "VI with NFs 9-5", "color": "#2ca02c", "alpha": 0.7},
        "ToyModelVINF_9and8": {"title": "VI with NFs 9-8", "color": "#17becf", "alpha": 0.6},
        "ToyModelVINF_9and11": {"title": "VI with NFs 9-11", "color": "#8c564b", "alpha": 0.5},
        "ToyModelVINF_12and5": {"title": "VI with NFs 12-5", "color": "#e377c2", "alpha": 0.4},
        "ToyModelVINF_12and8": {"title": "VI with NFs 12-8", "color": "#bcbd22", "alpha": 0.3},
        "ToyModelVINF_12and11": {"title": "VI with NFs 12-11", "color": "#9467bd", "alpha": 0.2},
    }
}


if __name__ == "__main__":
    steps = np.arange(1, N_SAMPLES + 1)

    args = setup_argparse()
    for prob, prob_dict in DEFAULT_DATA_DICT.items():
        for exp in range(args.start, args.end + 1):
            plot_running_mp_trace(prob, prob_dict, [exp], steps, N_SAMPLES, figsize=(12, 6), suffix="_ablation")
