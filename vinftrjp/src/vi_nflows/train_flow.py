import numpy as np
import torch
from normflows.flows import (
    CoupledRationalQuadraticSpline,
    MaskedAffineAutoregressive,
    MaskedAffineFlow,
    Planar,
)
from normflows.nets import MLP
from tqdm import tqdm

from .base import ConditionalNormalizingFlow, NormalizingFlow
from .core_distributions import DiagGaussian, DiagStudentT
from .coupling import ConditionalMaskedAffineFlow
from .new_flows import MaskedAffineFlowThresholded, PositiveConstraintLayer, TrainableLOFTLayer


class TrainNormalizingFlowBase:
    """Base trainer for normalizing flow models using reverse KL divergence optimization.

    Implements the core training loop for variational inference with normalizing flows,
    supporting both Gaussian and Student's t base distributions, annealing schedules,
    and early stopping mechanisms.

    Parameters
    ----------
    q0 : object, optional
        Base distribution for the flow. If None, will be initialized automatically
        based on other parameters.
    initial_loc_spec : str, optional
        Specification method for base distribution initialization. Determines how
        the initial location parameters are set when q0 is None.
    use_student_base : bool, default=False
        Whether to use Student's t-distribution as base distribution instead of
        Gaussian. Provides heavier tails for improved robustness.
    annealing : bool, default=False
        Whether to use temperature annealing during training. Gradually adjusts
        the temperature parameter to improve optimization stability.
    anneal_iter : int, optional
        Number of iterations for annealing schedule. Currently unused in implementation
        but reserved for future annealing strategies.
    lr : float, default=1e-4
        Learning rate for the optimizer. Controls step size in parameter updates.
    weight_decay : float, default=0.0
        Weight decay (L2 regularization penalty) for optimizer. Helps prevent
        overfitting by penalizing large parameter values.
    max_iter : int, default=10000
        Maximum number of training iterations. Training will stop after this many
        iterations unless early stopping criteria are met.
    num_samples : int, default=2**8
        Number of samples to draw per iteration for Monte Carlo gradient estimation.
        Recommended range is 64-512 for balance between variance and computation.
    device : str, default="cpu"
        Device to run training on. Can be 'cpu' for CPU computation or 'cuda' for
        GPU acceleration.
    save_path : str, optional
        File path for saving the best model checkpoint. If provided, model state
        will be saved when validation loss improves.
    verbose : bool, default=True
        Whether to print training progress and metrics during optimization.
    use_loft : bool, default=True
        Whether to use LOFT (Layer-wise Optimal Transport) layer in the flow
        architecture. Improves training stability and convergence.
    t : float, default=1
        Temperature parameter for annealing. Controls the sharpness of the target
        distribution during tempered transitions.
    patience : int, optional
        Number of iterations to wait for improvement before early stopping.
        If None, defaults to max_iter (no early stopping).
    improvement_threshold : float, optional
        Minimum improvement in loss required to reset patience counter.
        If None, defaults to 0.0 (any improvement resets patience).
    """

    def __init__(
        self,
        q0=None,
        initial_loc_spec=None,
        use_student_base=False,
        annealing=False,
        anneal_iter=None,  # Unused parameter; beta computed per iteration
        lr=1e-4,
        weight_decay=0.0,
        max_iter=10000,
        num_samples=2**8,  # Samples per iteration (recommended range: 64-512)
        device="cpu",
        save_path=None,
        verbose=True,
        use_loft=True,
        t=1,
        patience=None,
        improvement_threshold=None,
    ) -> None:
        # Base distribution configuration
        self.q0 = q0
        self.initial_loc_spec = initial_loc_spec  # For generating q0 when q0 is None
        self.use_student_base = use_student_base  # For generating q0 when q0 is None

        # Annealing control parameters
        self.annealing = annealing
        self.anneal_iter = anneal_iter

        # Optimization hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay

        # Training configuration parameters
        self.max_iter = max_iter
        self.num_samples = num_samples
        self.device = torch.device(device if torch.cuda.is_available() and device.lower().startswith("cuda") else "cpu")

        # Output and logging configuration
        self.save_path = save_path
        self.verbose = verbose

        # LOFT (Layer-wise Optimal Transport) configuration
        self.use_loft = use_loft

        # Network architecture hyperparameters
        self.hidden_layer_size = 256
        self.LOFT_THRESHOLD_VALUE = 100.0
        self.REAL_NVP_SCALING_FACTOR_BOUND_VALUE = 0.1
        self.REAL_NVP_SCALING_FACTOR_BOUND_TYPE = None

        # Temperature parameter for annealing (float)
        self.t = t

        # Early stopping configuration
        if patience is None:
            patience = max_iter  # Default to no early stopping
        self.patience = patience

        if improvement_threshold is None:
            improvement_threshold = 0.0  # Default to any improvement
        self.improvement_threshold = improvement_threshold

        self.clip_grad_max_norm = 1.0

    def _get_configs_from_target(self, latent_size, **kwargs):
        """Determine flow architecture based on problem dimensionality.

        Parameters
        ----------
        latent_size : int
            Dimensionality of the latent space.

        Returns
        -------
        flow_type : str
            Type of flow to use ('Planar' or 'RealNVP').
        num_of_flows : int
            Number of flow layers in the model.
        trainable_base : bool
            Whether the base distribution parameters are trainable.
        """
        raise NotImplementedError

    def run(self, target, **kwargs):
        """Execute the complete training pipeline.

        Parameters
        ----------
        target : object
            Target distribution to approximate.

        Returns
        -------
        model : NormalizingFlow
            Trained normalizing flow model.

        Raises
        ------
        ValueError
            If target distribution does not provide dimensionality information.
        """
        # Get latent dimension from target distribution
        if target is not None and hasattr(target, "ndim"):  # Check if target has 'ndim' attribute
            latent_size = target.ndim
        else:
            # Raise error if target distribution does not provide dimensionality information
            raise ValueError(f"Target distribution {target} does not have required 'ndim' attribute")

        # Get model configuration from target distribution
        flow_type, num_of_flows, trainable_base = self._get_configs_from_target(latent_size, **kwargs)

        # Initialize base distribution
        q0 = self.initiate_distribution(latent_size, trainable_base)

        # Construct flow transformations
        flows = self.construct_flows(latent_size, flow_type, num_of_flows, trainable_base, target=target)

        # Build complete model and optimizer
        model, optimizer, scheduler = self.build_model(q0, flows, target)

        # Train the normalizing flow model
        model = self.train_model(model, optimizer, scheduler)

        return model

    def initiate_distribution(self, latent_size, trainable_base):
        """Initialize the base distribution for the flow model.

        Parameters
        ----------
        latent_size : int
            Dimensionality of the latent space.
        trainable_base : bool
            Whether base distribution parameters should be trainable.

        Returns
        -------
        q0 : object
            Initialized base distribution.

        Raises
        ------
        ValueError
            If initial_loc_spec is not supported.
        """
        if self.q0 is not None:
            return self.q0

        return self._initiate_distribution(latent_size, trainable_base)

    def _initiate_distribution(self, latent_size, trainable_base):
        raise NotImplementedError

    def construct_flows(self, latent_size, flow_type, num_of_flows, trainable_base, target=None):
        """Construct the sequence of flow transformations.

        Parameters
        ----------
        latent_size : int
            Dimensionality of the latent space.
        flow_type : str
            Type of flow architecture to use.
        num_of_flows : int
            Number of flow layers to create.
        trainable_base : bool
            Whether base distribution is trainable.

        Returns
        -------
        flows : list
            List of flow transformation layers.

        Raises
        ------
        ValueError
            If flow_type is not supported.
        """
        # Initialize empty list for flow layers
        flows = []

        # Create binary mask for coupling layers
        # Alternating pattern: [1, 0, 1, 0, ...] for splitting dimensions
        binary_mask = torch.tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)], device=self.device)

        for i in range(num_of_flows):
            flows += self._construct_flow(num_of_flows, i, binary_mask, flow_type, latent_size, target)

        if target is not None and hasattr(target, "get_pos_constraint_ids"):
            pos_constraint_ids = target.get_pos_constraint_ids()
            flows += [PositiveConstraintLayer(pos_constraint_ids, latent_size)]

        return flows

    def _construct_flow(self, num_of_flows, index_flow, binary_mask, flow_type, latent_size, target):
        raise NotImplementedError

    def build_model(self, q0, flows, target):
        """Construct the complete model and optimizer.

        Parameters
        ----------
        q0 : object
            Base distribution.
        flows : list
            List of flow transformation layers.
        target : object
            Target distribution.

        Returns
        -------
        model : NormalizingFlow
            Complete normalizing flow model.
        optimizer : torch.optim.Optimizer
            Adam optimizer for training.
        """
        # Construct the model
        model = self._build_model(q0=q0, flows=flows, target=target)

        # Initialize Adam optimizer with specified learning parameters
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=self.patience // 10, factor=0.5, threshold=self.improvement_threshold
        )

        return model, optimizer, scheduler

    def _build_model(self, q0, flows, target):
        raise NotImplementedError

    def train_model(self, model, optimizer, scheduler):
        """Execute the training loop with reverse KL divergence minimization.

        Parameters
        ----------
        model : NormalizingFlow
            The normalizing flow model to train.
        optimizer : torch.optim.Optimizer
            Optimizer for updating model parameters.

        Returns
        -------
        model : NormalizingFlow
            Trained normalizing flow model.
        """
        # Training state variables
        best_loss = float("inf")  # Initialize with infinity
        best_state = None  # Store best model parameters
        patience_counter = 0

        # Training loop with progress bar
        pbar = tqdm(range(self.max_iter), disable=not self.verbose)
        for it in pbar:
            optimizer.zero_grad()  # Reset gradients

            try:
                model, beta, loss = self._train_model(model, it)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clip_grad_max_norm)

                # Check for invalid gradients (NaN or Inf values)
                invalid_grad = self._check_gradient_validity(model)

                # Update parameters only if gradients are numerically valid
                if not invalid_grad:
                    optimizer.step()
                    scheduler.step(loss.detach().item())
                elif self.verbose:
                    print(f"Iteration {it}: Invalid gradient detected - update skipped")

                # Extract loss value
                loss_val = loss.item()

                # Save best model and update loss
                patience_counter, best_loss, best_state = self._save_model_update_loss(
                    model, loss_val, best_loss, best_state, patience_counter, it
                )

                # Update progress bar with current loss and beta
                if self.verbose:
                    pbar.set_postfix({"loss": f"{loss_val:.4f}", "best": f"{best_loss:.4f}", "beta": f"{beta:.3f}"})

                # Early stopping
                if patience_counter >= self.patience:
                    print(f"Early stopping at loss: {best_loss}")
                    break

            except (RuntimeError, ValueError) as e:
                # Handle any exceptions during training iteration
                if self.verbose:
                    print(f"Iteration {it} failed: {e}")

                continue

        # Load best model parameters after training
        if best_state is not None:
            model.load_state_dict(best_state)

        # Print final training summary
        if self.verbose:
            print(f"Training completed. Best loss: {best_loss:.6f}")

        return model

    def _train_model(self, model, it):
        raise NotImplementedError

    def _compute_annealing_beta(self, iteration):
        # Compute annealing beta parameter
        if self.annealing:
            if (self.anneal_iter is not None) and (self.anneal_iter > 0):
                # Linear annealing from 0.01 to 1.0 over anneal_iter iterations
                beta = min(1.0, 0.01 + iteration / self.anneal_iter)
            else:
                beta = min(1.0, 0.01 + iteration / self.max_iter)
        else:
            # No annealing: fixed beta = 1.0
            beta = 1.0
        return beta

    def _check_gradient_validity(self, model):
        """Check if gradients are numerically valid."""
        for param in model.parameters():
            if param.grad is not None:
                grad = param.grad
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    return True
        return False

    def _save_model_update_loss(self, model, loss_val, best_loss, best_state, patience_counter, iteration):
        # Save best model if current loss improves and is numerically valid
        if not (np.isnan(loss_val) or np.isinf(loss_val)):
            threshold = min(self.improvement_threshold, self.improvement_threshold * abs(best_loss))
            if loss_val < best_loss - threshold:
                best_loss = loss_val
                best_state = model.state_dict()
                if self.save_path:
                    torch.save(best_state, self.save_path)

                patience_counter = 0
            else:
                patience_counter += 1

        else:
            patience_counter += 1
            if self.verbose:
                print(f"Iteration {iteration}: Invalid loss value - update skipped")

        return patience_counter, best_loss, best_state


class TrainNormalizingFlow(TrainNormalizingFlowBase):
    """Trainer for normalizing flow models using reverse KL divergence."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _get_configs_from_target(self, latent_size, **kwargs):
        """Determine flow architecture based on problem dimensionality.

        Parameters
        ----------
        latent_size : int
            Dimensionality of the latent space.

        Returns
        -------
        flow_type : str
            Type of flow to use ('Planar' or 'RealNVP').
        num_of_flows : int
            Number of flow layers in the model.
        trainable_base : bool
            Whether the base distribution parameters are trainable.
        """
        # Determine flow type based on problem dimensionality
        if latent_size == 1:
            flow_type = kwargs.get("flow_type_1d", "Planar")
            num_of_flows = kwargs.get("num_of_flows_1d", 9)
            trainable_base = kwargs.get("trainable_base_1d", False)
        else:
            flow_type = kwargs.get("flow_type_2d", "RealNVP")
            num_of_flows = kwargs.get("num_of_flows_2d", 8)
            trainable_base = kwargs.get("trainable_base_2d", False)

        return flow_type, num_of_flows, trainable_base

    def _initiate_distribution(self, latent_size, trainable_base):
        # Initialize base distribution location parameter
        if self.initial_loc_spec == "random_large":
            # Large random initialization: uniform distribution over [-10, 10]
            initial_loc_base = torch.rand(size=(1, latent_size)) * 20.0 - 10.0
        elif self.initial_loc_spec == "random_small":
            # Small random initialization: uniform distribution over [-1, 1]
            initial_loc_base = torch.rand(size=(1, latent_size)) * 2.0 - 1.0
        elif self.initial_loc_spec == "zeros":
            # Zero initialization: all zeros
            initial_loc_base = torch.zeros(size=(1, latent_size))
        else:
            raise ValueError(f"Unsupported initial_loc_spec: {self.initial_loc_spec}")

        # Create base distribution (Gaussian or Student's t)
        if self.use_student_base:
            # Student's t-distribution with heavier tails
            q0 = DiagStudentT(latent_size, initial_loc=initial_loc_base, trainable=trainable_base)
        else:
            # Standard Gaussian distribution
            q0 = DiagGaussian(latent_size, initial_loc=initial_loc_base, trainable=trainable_base)

        return q0

    def _construct_flow(self, num_of_flows, index_flow, binary_mask, flow_type, latent_size, target):
        flows = []
        if flow_type in ["RealNVP", "RealNVP_1d"]:
            # RealNVP: Non-volume preserving flow with coupling layers
            # MLP networks for scale and translation transformations
            scale_nn = MLP([latent_size, self.hidden_layer_size, latent_size], init_zeros=True)
            translation_nn = MLP([latent_size, self.hidden_layer_size, latent_size], init_zeros=True)

            # Alternate mask pattern for consecutive layers
            current_mask = binary_mask if index_flow % 2 == 0 else 1 - binary_mask

            if flow_type == "RealNVP" and self.REAL_NVP_SCALING_FACTOR_BOUND_TYPE is not None:
                # Use thresholded version if bounds are specified
                flow = MaskedAffineFlowThresholded(
                    current_mask,
                    translation_nn,
                    scale_nn,
                    self.REAL_NVP_SCALING_FACTOR_BOUND_VALUE,
                    self.REAL_NVP_SCALING_FACTOR_BOUND_TYPE,
                )
            else:
                # Standard RealNVP coupling layer
                flow = MaskedAffineFlow(current_mask, translation_nn, scale_nn)
            flows.append(flow)

            # Add LOFT layer at the end of flow stack if enabled
            if (index_flow == num_of_flows - 1) and self.use_loft and (latent_size > 1):
                flow = TrainableLOFTLayer(latent_size, self.LOFT_THRESHOLD_VALUE, train_t=False)
                flows.append(flow)

        elif flow_type == "MAF":
            # Masked Autoregressive Flow: autoregressive transformations
            flow = MaskedAffineAutoregressive(latent_size, 2 * latent_size)
            flows.append(flow)

        elif flow_type == "Planar":
            # Planar flow: simple but expressive single-layer transformation
            flow = Planar((latent_size,), act="leaky_relu")
            flows.append(flow)

        elif flow_type == "NeuralSpline":
            # Neural Spline Flow: flexible spline-based transformations
            flow = CoupledRationalQuadraticSpline(
                latent_size, 1, self.hidden_layer_size, reverse_mask=(index_flow % 2 == 0)
            )
            flows.append(flow)

        else:
            raise ValueError(f"Unsupported flow_type: {flow_type}")

        return flows

    def _build_model(self, q0, flows, target):
        # Construct the complete normalizing flow model
        return NormalizingFlow(q0=q0, flows=flows, p=target).to(self.device)

    def _train_model(self, model, it):
        # Compute annealing beta parameter
        beta = self._compute_annealing_beta(it)

        # Compute reverse KL divergence loss
        loss = model.reverse_kld(num_samples=self.num_samples, beta=beta)
        loss.backward()  # Backpropagate gradients
        return model, beta, loss


class TrainConditionalNormalizingFlow(TrainNormalizingFlow):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _construct_flow(self, num_of_flows, index_flow, binary_mask, flow_type, latent_size, target):
        flows = []
        if flow_type in ["RealNVP"]:
            # RealNVP: Non-volume preserving flow with coupling layers
            # MLP networks for scale and translation transformations
            # [z_masked, context]
            scale_nn = MLP([latent_size + latent_size - 1, self.hidden_layer_size, latent_size], init_zeros=True)
            translation_nn = MLP([latent_size + latent_size - 1, self.hidden_layer_size, latent_size], init_zeros=True)

            # Alternate mask pattern for consecutive layers
            current_mask = binary_mask if index_flow % 2 == 0 else 1 - binary_mask

            # Conditional RealNVP coupling layer
            flow = ConditionalMaskedAffineFlow(current_mask, translation_nn, scale_nn)
            flows.append(flow)

        else:
            raise ValueError(f"Unsupported flow_type: {flow_type}")

        return flows

    def _build_model(self, q0, flows, target):
        # Construct the complete conditional normalizing flow model
        return ConditionalNormalizingFlow(q0=q0, flows=flows, p=target).to(self.device)

    def _train_model(self, model, it):
        # Compute annealing beta parameter
        beta = self._compute_annealing_beta(it)

        context = self._generate_random_context(self.num_samples)

        # Compute reverse KL divergence loss
        loss = model.reverse_kld(num_samples=self.num_samples, beta=beta, context=context)
        loss.backward()  # Backpropagate gradients
        return model, beta, loss

    def _generate_random_context(self, batch_size: int):
        masks = torch.tensor([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
        context = masks[torch.randint(0, len(masks), (batch_size,))].float()
        return context.to(self.device)
