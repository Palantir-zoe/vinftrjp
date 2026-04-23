from src.vi_nflows import TrainConditionalNormalizingFlow, TrainNormalizingFlow


def build_normalizing_flows(*, max_iter=20000, lr=5e-5, num_samples=2**8, device="cpu", verbose=True):
    def _train(q0=None, target=None):
        config = {
            "flow_type_1d": "Planar",
            "num_of_flows_1d": 16,
            "trainable_base_1d": True,
            "flow_type_2d": "RealNVP",
            "num_of_flows_2d": 16,
            "trainable_base_2d": True,
        }
        if hasattr(target, "sample_context"):
            config["flow_type_1d"] = "RealNVP"
        trainer_class = TrainConditionalNormalizingFlow if hasattr(target, "sample_context") else TrainNormalizingFlow
        trainer = trainer_class(
            q0=q0,
            initial_loc_spec="zeros",
            annealing=True,
            anneal_iter=max(1, int(0.8 * max_iter)),
            max_iter=max_iter,
            lr=lr,
            num_samples=num_samples,
            device=device,
            verbose=verbose,
        )
        # Change-point targets can produce very sharp posterior geometry near
        # simplex boundaries, so we use bounded RealNVP scales for stability.
        trainer.REAL_NVP_SCALING_FACTOR_BOUND_TYPE = "tanh"
        return trainer.run(target=target, **config)

    return _train
