from src.vi_nflows import TrainNormalizingFlow, resolve_flow_training_device


def _run_flow(q0, target, *, num_flows_1d, num_flows_2d, device=None):
    kwargs = {
        "max_iter": 10000,
        "lr": 1e-4,
        "num_samples": 2**8,
        "device": resolve_flow_training_device(device),
        "verbose": True,
    }
    config = {
        "flow_type_1d": "Planar",
        "num_of_flows_1d": num_flows_1d,
        "trainable_base_1d": False,
        "flow_type_2d": "RealNVP",
        "num_of_flows_2d": num_flows_2d,
        "trainable_base_2d": False,
    }
    return TrainNormalizingFlow(q0=q0, **kwargs).run(target=target, **config)


def _make_flow_builder(name, num_flows_1d, num_flows_2d):
    def _builder(q0, target, *, device=None):
        return _run_flow(
            q0,
            target,
            num_flows_1d=num_flows_1d,
            num_flows_2d=num_flows_2d,
            device=device,
        )

    _builder.__name__ = name
    return _builder


get_normalizing_flows = _make_flow_builder("get_normalizing_flows", 9, 8)

for _num_flows_1d, _num_flows_2d in [(6, 5), (6, 8), (6, 11), (9, 5), (9, 8), (9, 11), (12, 5), (12, 8), (12, 11)]:
    _name = f"get_normalizing_flows_{_num_flows_1d}_{_num_flows_2d}"
    globals()[_name] = _make_flow_builder(_name, _num_flows_1d, _num_flows_2d)
