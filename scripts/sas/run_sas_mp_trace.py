from scripts.sas.sas_base import N_SAMPLES, configure_flow_training, setup_argparse
from scripts.sas.sas_vinfs import get_normalizing_flows
from src.main import Experiments
from src.vi_nflows import resolve_flow_training_device

PROBLEMS = ["SAS"]
ALGORITHMS = ["ToyModelAF", "ToyModelNF", "ToyModelPerfect", "ToyModelVINF"]


if __name__ == "__main__":
    args = setup_argparse()
    configure_flow_training(
        args.flow_device,
        flow_num_samples=args.flow_num_samples,
        flow_hidden_layer_size=args.flow_hidden_layer_size,
        flow_annealing=args.flow_annealing,
    )
    run_device = "cuda" if resolve_flow_training_device(args.flow_device).startswith("cuda") else "cpu"

    e = Experiments(args.start, args.end, problems=PROBLEMS, algorithms=ALGORITHMS, device=run_device)
    e.run(n_samples=N_SAMPLES, normalizing_flows=get_normalizing_flows, save_flows_dir=args.save_flows_dir)
