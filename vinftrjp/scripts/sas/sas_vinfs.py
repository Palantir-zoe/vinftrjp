from src.vi_nflows import TrainNormalizingFlow


def get_normalizing_flows(q0, target):
    kwargs = {"max_iter": 10000, "lr": 1e-4, "num_samples": 2**8, "device": "cpu", "verbose": True}
    config = {
        "flow_type_1d": "Planar",
        "num_of_flows_1d": 9,
        "trainable_base_1d": False,
        "flow_type_2d": "RealNVP",
        "num_of_flows_2d": 8,
        "trainable_base_2d": False,
    }
    model = TrainNormalizingFlow(q0=q0, **kwargs).run(target=target, **config)
    return model


def get_normalizing_flows_6_5(q0, target):
    kwargs = {"max_iter": 10000, "lr": 1e-4, "num_samples": 2**8, "device": "cpu", "verbose": True}
    config = {
        "flow_type_1d": "Planar",
        "num_of_flows_1d": 6,
        "trainable_base_1d": False,
        "flow_type_2d": "RealNVP",
        "num_of_flows_2d": 5,
        "trainable_base_2d": False,
    }
    model = TrainNormalizingFlow(q0=q0, **kwargs).run(target=target, **config)
    return model


def get_normalizing_flows_6_8(q0, target):
    kwargs = {"max_iter": 10000, "lr": 1e-4, "num_samples": 2**8, "device": "cpu", "verbose": True}
    config = {
        "flow_type_1d": "Planar",
        "num_of_flows_1d": 6,
        "trainable_base_1d": False,
        "flow_type_2d": "RealNVP",
        "num_of_flows_2d": 8,
        "trainable_base_2d": False,
    }
    model = TrainNormalizingFlow(q0=q0, **kwargs).run(target=target, **config)
    return model


def get_normalizing_flows_6_11(q0, target):
    kwargs = {"max_iter": 10000, "lr": 1e-4, "num_samples": 2**8, "device": "cpu", "verbose": True}
    config = {
        "flow_type_1d": "Planar",
        "num_of_flows_1d": 6,
        "trainable_base_1d": False,
        "flow_type_2d": "RealNVP",
        "num_of_flows_2d": 11,
        "trainable_base_2d": False,
    }
    model = TrainNormalizingFlow(q0=q0, **kwargs).run(target=target, **config)
    return model


def get_normalizing_flows_9_5(q0, target):
    kwargs = {"max_iter": 10000, "lr": 1e-4, "num_samples": 2**8, "device": "cpu", "verbose": True}
    config = {
        "flow_type_1d": "Planar",
        "num_of_flows_1d": 9,
        "trainable_base_1d": False,
        "flow_type_2d": "RealNVP",
        "num_of_flows_2d": 5,
        "trainable_base_2d": False,
    }
    model = TrainNormalizingFlow(q0=q0, **kwargs).run(target=target, **config)
    return model


def get_normalizing_flows_9_8(q0, target):
    kwargs = {"max_iter": 10000, "lr": 1e-4, "num_samples": 2**8, "device": "cpu", "verbose": True}
    config = {
        "flow_type_1d": "Planar",
        "num_of_flows_1d": 9,
        "trainable_base_1d": False,
        "flow_type_2d": "RealNVP",
        "num_of_flows_2d": 8,
        "trainable_base_2d": False,
    }
    model = TrainNormalizingFlow(q0=q0, **kwargs).run(target=target, **config)
    return model


def get_normalizing_flows_9_11(q0, target):
    kwargs = {"max_iter": 10000, "lr": 1e-4, "num_samples": 2**8, "device": "cpu", "verbose": True}
    config = {
        "flow_type_1d": "Planar",
        "num_of_flows_1d": 9,
        "trainable_base_1d": False,
        "flow_type_2d": "RealNVP",
        "num_of_flows_2d": 11,
        "trainable_base_2d": False,
    }
    model = TrainNormalizingFlow(q0=q0, **kwargs).run(target=target, **config)
    return model


def get_normalizing_flows_12_5(q0, target):
    kwargs = {"max_iter": 10000, "lr": 1e-4, "num_samples": 2**8, "device": "cpu", "verbose": True}
    config = {
        "flow_type_1d": "Planar",
        "num_of_flows_1d": 12,
        "trainable_base_1d": False,
        "flow_type_2d": "RealNVP",
        "num_of_flows_2d": 5,
        "trainable_base_2d": False,
    }
    model = TrainNormalizingFlow(q0=q0, **kwargs).run(target=target, **config)
    return model


def get_normalizing_flows_12_8(q0, target):
    kwargs = {"max_iter": 10000, "lr": 1e-4, "num_samples": 2**8, "device": "cpu", "verbose": True}
    config = {
        "flow_type_1d": "Planar",
        "num_of_flows_1d": 12,
        "trainable_base_1d": False,
        "flow_type_2d": "RealNVP",
        "num_of_flows_2d": 8,
        "trainable_base_2d": False,
    }
    model = TrainNormalizingFlow(q0=q0, **kwargs).run(target=target, **config)
    return model


def get_normalizing_flows_12_11(q0, target):
    kwargs = {"max_iter": 10000, "lr": 1e-4, "num_samples": 2**8, "device": "cpu", "verbose": True}
    config = {
        "flow_type_1d": "Planar",
        "num_of_flows_1d": 12,
        "trainable_base_1d": False,
        "flow_type_2d": "RealNVP",
        "num_of_flows_2d": 11,
        "trainable_base_2d": False,
    }
    model = TrainNormalizingFlow(q0=q0, **kwargs).run(target=target, **config)
    return model
