from src.vi_nflows import TrainNormalizingFlow


def get_normalizing_flows(q0, target):
    kwargs = {"max_iter": 10000, "lr": 1e-4, "num_samples": 2**8, "device": "cpu", "verbose": True}
    config = {
        "flow_type_1d": "RealNVP",
        "num_of_flows_1d": 16,
        "trainable_base_1d": False,
        "flow_type_2d": "RealNVP",
        "num_of_flows_2d": 16,
        "trainable_base_2d": False,
    }
    model = TrainNormalizingFlow(q0=q0, **kwargs).run(target=target, **config)
    return model


def get_normalizing_flows_2(q0, target):
    kwargs = {"max_iter": 10000, "lr": 1e-4, "num_samples": 2**8, "device": "cpu", "verbose": True}
    config = {
        "flow_type_1d": "RealNVP",
        "num_of_flows_1d": 16,
        "trainable_base_1d": False,
        "flow_type_2d": "RealNVP",
        "num_of_flows_2d": 2,
        "trainable_base_2d": False,
    }
    model = TrainNormalizingFlow(q0=q0, **kwargs).run(target=target, **config)
    return model


def get_normalizing_flows_4(q0, target):
    kwargs = {"max_iter": 10000, "lr": 1e-4, "num_samples": 2**8, "device": "cpu", "verbose": True}
    config = {
        "flow_type_1d": "RealNVP",
        "num_of_flows_1d": 16,
        "trainable_base_1d": False,
        "flow_type_2d": "RealNVP",
        "num_of_flows_2d": 4,
        "trainable_base_2d": False,
    }
    model = TrainNormalizingFlow(q0=q0, **kwargs).run(target=target, **config)
    return model


def get_normalizing_flows_6(q0, target):
    kwargs = {"max_iter": 10000, "lr": 1e-4, "num_samples": 2**8, "device": "cpu", "verbose": True}
    config = {
        "flow_type_1d": "RealNVP",
        "num_of_flows_1d": 16,
        "trainable_base_1d": False,
        "flow_type_2d": "RealNVP",
        "num_of_flows_2d": 6,
        "trainable_base_2d": False,
    }
    model = TrainNormalizingFlow(q0=q0, **kwargs).run(target=target, **config)
    return model


def get_normalizing_flows_8(q0, target):
    kwargs = {"max_iter": 10000, "lr": 1e-4, "num_samples": 2**8, "device": "cpu", "verbose": True}
    config = {
        "flow_type_1d": "RealNVP",
        "num_of_flows_1d": 16,
        "trainable_base_1d": False,
        "flow_type_2d": "RealNVP",
        "num_of_flows_2d": 8,
        "trainable_base_2d": False,
    }
    model = TrainNormalizingFlow(q0=q0, **kwargs).run(target=target, **config)
    return model


def get_normalizing_flows_10(q0, target):
    kwargs = {"max_iter": 10000, "lr": 1e-4, "num_samples": 2**8, "device": "cpu", "verbose": True}
    config = {
        "flow_type_1d": "RealNVP",
        "num_of_flows_1d": 16,
        "trainable_base_1d": False,
        "flow_type_2d": "RealNVP",
        "num_of_flows_2d": 10,
        "trainable_base_2d": False,
    }
    model = TrainNormalizingFlow(q0=q0, **kwargs).run(target=target, **config)
    return model


def get_normalizing_flows_12(q0, target):
    kwargs = {"max_iter": 10000, "lr": 1e-4, "num_samples": 2**8, "device": "cpu", "verbose": True}
    config = {
        "flow_type_1d": "RealNVP",
        "num_of_flows_1d": 16,
        "trainable_base_1d": False,
        "flow_type_2d": "RealNVP",
        "num_of_flows_2d": 12,
        "trainable_base_2d": False,
    }
    model = TrainNormalizingFlow(q0=q0, **kwargs).run(target=target, **config)
    return model


def get_normalizing_flows_14(q0, target):
    kwargs = {"max_iter": 10000, "lr": 1e-4, "num_samples": 2**8, "device": "cpu", "verbose": True}
    config = {
        "flow_type_1d": "RealNVP",
        "num_of_flows_1d": 16,
        "trainable_base_1d": False,
        "flow_type_2d": "RealNVP",
        "num_of_flows_2d": 14,
        "trainable_base_2d": False,
    }
    model = TrainNormalizingFlow(q0=q0, **kwargs).run(target=target, **config)
    return model


def get_normalizing_flows_16(q0, target):
    kwargs = {"max_iter": 10000, "lr": 1e-4, "num_samples": 2**8, "device": "cpu", "verbose": True}
    config = {
        "flow_type_1d": "RealNVP",
        "num_of_flows_1d": 16,
        "trainable_base_1d": False,
        "flow_type_2d": "RealNVP",
        "num_of_flows_2d": 16,
        "trainable_base_2d": False,
    }
    model = TrainNormalizingFlow(q0=q0, **kwargs).run(target=target, **config)
    return model


def get_normalizing_flows_18(q0, target):
    kwargs = {"max_iter": 10000, "lr": 1e-4, "num_samples": 2**8, "device": "cpu", "verbose": True}
    config = {
        "flow_type_1d": "RealNVP",
        "num_of_flows_1d": 16,
        "trainable_base_1d": False,
        "flow_type_2d": "RealNVP",
        "num_of_flows_2d": 18,
        "trainable_base_2d": False,
    }
    model = TrainNormalizingFlow(q0=q0, **kwargs).run(target=target, **config)
    return model


def get_normalizing_flows_20(q0, target):
    kwargs = {"max_iter": 10000, "lr": 1e-4, "num_samples": 2**8, "device": "cpu", "verbose": True}
    config = {
        "flow_type_1d": "RealNVP",
        "num_of_flows_1d": 16,
        "trainable_base_1d": False,
        "flow_type_2d": "RealNVP",
        "num_of_flows_2d": 20,
        "trainable_base_2d": False,
    }
    model = TrainNormalizingFlow(q0=q0, **kwargs).run(target=target, **config)
    return model


def get_normalizing_flows_22(q0, target):
    kwargs = {"max_iter": 10000, "lr": 1e-4, "num_samples": 2**8, "device": "cpu", "verbose": True}
    config = {
        "flow_type_1d": "RealNVP",
        "num_of_flows_1d": 16,
        "trainable_base_1d": False,
        "flow_type_2d": "RealNVP",
        "num_of_flows_2d": 22,
        "trainable_base_2d": False,
    }
    model = TrainNormalizingFlow(q0=q0, **kwargs).run(target=target, **config)
    return model


def get_normalizing_flows_24(q0, target):
    kwargs = {"max_iter": 10000, "lr": 1e-4, "num_samples": 2**8, "device": "cpu", "verbose": True}
    config = {
        "flow_type_1d": "RealNVP",
        "num_of_flows_1d": 16,
        "trainable_base_1d": False,
        "flow_type_2d": "RealNVP",
        "num_of_flows_2d": 24,
        "trainable_base_2d": False,
    }
    model = TrainNormalizingFlow(q0=q0, **kwargs).run(target=target, **config)
    return model


def get_normalizing_flows_26(q0, target):
    kwargs = {"max_iter": 10000, "lr": 1e-4, "num_samples": 2**8, "device": "cpu", "verbose": True}
    config = {
        "flow_type_1d": "RealNVP",
        "num_of_flows_1d": 16,
        "trainable_base_1d": False,
        "flow_type_2d": "RealNVP",
        "num_of_flows_2d": 26,
        "trainable_base_2d": False,
    }
    model = TrainNormalizingFlow(q0=q0, **kwargs).run(target=target, **config)
    return model


def get_normalizing_flows_28(q0, target):
    kwargs = {"max_iter": 10000, "lr": 1e-4, "num_samples": 2**8, "device": "cpu", "verbose": True}
    config = {
        "flow_type_1d": "RealNVP",
        "num_of_flows_1d": 16,
        "trainable_base_1d": False,
        "flow_type_2d": "RealNVP",
        "num_of_flows_2d": 28,
        "trainable_base_2d": False,
    }
    model = TrainNormalizingFlow(q0=q0, **kwargs).run(target=target, **config)
    return model


def get_normalizing_flows_30(q0, target):
    kwargs = {"max_iter": 10000, "lr": 1e-4, "num_samples": 2**8, "device": "cpu", "verbose": True}
    config = {
        "flow_type_1d": "RealNVP",
        "num_of_flows_1d": 16,
        "trainable_base_1d": False,
        "flow_type_2d": "RealNVP",
        "num_of_flows_2d": 30,
        "trainable_base_2d": False,
    }
    model = TrainNormalizingFlow(q0=q0, **kwargs).run(target=target, **config)
    return model


def get_normalizing_flows_32(q0, target):
    kwargs = {"max_iter": 10000, "lr": 1e-4, "num_samples": 2**8, "device": "cpu", "verbose": True}
    config = {
        "flow_type_1d": "RealNVP",
        "num_of_flows_1d": 16,
        "trainable_base_1d": False,
        "flow_type_2d": "RealNVP",
        "num_of_flows_2d": 32,
        "trainable_base_2d": False,
    }
    model = TrainNormalizingFlow(q0=q0, **kwargs).run(target=target, **config)
    return model


def get_normalizing_flows_34(q0, target):
    kwargs = {"max_iter": 10000, "lr": 1e-4, "num_samples": 2**8, "device": "cpu", "verbose": True}
    config = {
        "flow_type_1d": "RealNVP",
        "num_of_flows_1d": 16,
        "trainable_base_1d": False,
        "flow_type_2d": "RealNVP",
        "num_of_flows_2d": 34,
        "trainable_base_2d": False,
    }
    model = TrainNormalizingFlow(q0=q0, **kwargs).run(target=target, **config)
    return model


def get_normalizing_flows_36(q0, target):
    kwargs = {"max_iter": 10000, "lr": 1e-4, "num_samples": 2**8, "device": "cpu", "verbose": True}
    config = {
        "flow_type_1d": "RealNVP",
        "num_of_flows_1d": 16,
        "trainable_base_1d": False,
        "flow_type_2d": "RealNVP",
        "num_of_flows_2d": 36,
        "trainable_base_2d": False,
    }
    model = TrainNormalizingFlow(q0=q0, **kwargs).run(target=target, **config)
    return model


def get_normalizing_flows_38(q0, target):
    kwargs = {"max_iter": 10000, "lr": 1e-4, "num_samples": 2**8, "device": "cpu", "verbose": True}
    config = {
        "flow_type_1d": "RealNVP",
        "num_of_flows_1d": 16,
        "trainable_base_1d": False,
        "flow_type_2d": "RealNVP",
        "num_of_flows_2d": 38,
        "trainable_base_2d": False,
    }
    model = TrainNormalizingFlow(q0=q0, **kwargs).run(target=target, **config)
    return model


def get_normalizing_flows_40(q0, target):
    kwargs = {"max_iter": 10000, "lr": 1e-4, "num_samples": 2**8, "device": "cpu", "verbose": True}
    config = {
        "flow_type_1d": "RealNVP",
        "num_of_flows_1d": 16,
        "trainable_base_1d": False,
        "flow_type_2d": "RealNVP",
        "num_of_flows_2d": 40,
        "trainable_base_2d": False,
    }
    model = TrainNormalizingFlow(q0=q0, **kwargs).run(target=target, **config)
    return model


def get_normalizing_flows_42(q0, target):
    kwargs = {"max_iter": 10000, "lr": 1e-4, "num_samples": 2**8, "device": "cpu", "verbose": True}
    config = {
        "flow_type_1d": "RealNVP",
        "num_of_flows_1d": 16,
        "trainable_base_1d": False,
        "flow_type_2d": "RealNVP",
        "num_of_flows_2d": 12,
        "trainable_base_2d": False,
    }
    model = TrainNormalizingFlow(q0=q0, **kwargs).run(target=target, **config)
    return model


def get_normalizing_flows_44(q0, target):
    kwargs = {"max_iter": 10000, "lr": 1e-4, "num_samples": 2**8, "device": "cpu", "verbose": True}
    config = {
        "flow_type_1d": "RealNVP",
        "num_of_flows_1d": 16,
        "trainable_base_1d": False,
        "flow_type_2d": "RealNVP",
        "num_of_flows_2d": 44,
        "trainable_base_2d": False,
    }
    model = TrainNormalizingFlow(q0=q0, **kwargs).run(target=target, **config)
    return model
