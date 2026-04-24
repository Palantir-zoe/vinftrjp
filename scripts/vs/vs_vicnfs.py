from src.vi_nflows import TrainConditionalNormalizingFlow, resolve_flow_training_device


def _run_flow(q0, target, *, num_flows_2d, device=None):
    kwargs = {
        "max_iter": 10000 * 4,
        "lr": 1e-4,
        "num_samples": 2**8 * 4,
        "device": resolve_flow_training_device(device),
        "verbose": True,
    }
    config = {
        "flow_type_1d": "RealNVP",
        "num_of_flows_1d": 16,
        "trainable_base_1d": False,
        "flow_type_2d": "RealNVP",
        "num_of_flows_2d": num_flows_2d,
        "trainable_base_2d": False,
    }
    return TrainConditionalNormalizingFlow(q0=q0, **kwargs).run(target=target, **config)


def _make_flow_builder(name, num_flows_2d):
    def _builder(q0, target, *, device=None):
        return _run_flow(q0, target, num_flows_2d=num_flows_2d, device=device)

    _builder.__name__ = name
    return _builder


get_normalizing_flows = _make_flow_builder("get_normalizing_flows", 16)

for _depth in range(2, 46, 2):
    globals()[f"get_normalizing_flows_{_depth}"] = _make_flow_builder(f"get_normalizing_flows_{_depth}", _depth)
