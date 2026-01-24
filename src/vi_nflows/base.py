import torch
import torch.nn as nn

from .utils import set_requires_grad


# Modified from normflows.flows.base.Flow
class Flow(nn.Module):
    """
    Base class for normalizing flow transformations.
    """

    def __init__(self):
        super().__init__()

    def forward(self, z, context=None):
        """
        Transform input through the flow.

        Parameters
        ----------
        z : torch.Tensor
            Input tensor of shape (batch_size, ...).
        context : torch.Tensor, optional
            Context tensor for conditional flows, by default None.

        Returns
        -------
        torch.Tensor
            Transformed tensor of same shape as input.
        torch.Tensor
            Log absolute determinant of shape (batch_size,).

        Raises
        ------
        NotImplementedError
            If forward method is not implemented.
        """
        raise NotImplementedError("Forward pass has not been implemented.")

    def inverse(self, z, context=None):
        """
        Inverse transform through the flow.

        Parameters
        ----------
        z : torch.Tensor
            Input tensor of shape (batch_size, ...).
        context : torch.Tensor, optional
            Context tensor for conditional flows, by default None.

        Returns
        -------
        torch.Tensor
            Inverse transformed tensor.
        torch.Tensor
            Log absolute determinant of inverse transformation.

        Raises
        ------
        NotImplementedError
            If inverse method is not implemented.
        """
        raise NotImplementedError("This flow has no algebraic inverse.")


# Modified from normflows.flows.base.Composite
class Composite(Flow):
    """
    Composite flow combining multiple flows in sequence.
    """

    def __init__(self, flows):
        """
        Initialize composite flow.

        Parameters
        ----------
        flows : Iterable[Flow]
            Sequence of flow transformations to apply in order.
        """
        super().__init__()
        self._flows = nn.ModuleList(flows)

    @staticmethod
    def _cascade(z, funcs, context):
        """
        Apply sequence of transformations to input.

        Parameters
        ----------
        z : torch.Tensor
            Input tensor of shape (batch_size, ...).
        funcs : Iterable[Callable]
            Sequence of transformation functions.
        context : torch.Tensor, optional
            Context tensor for conditional flows, by default None.

        Returns
        -------
        torch.Tensor
            Transformed tensor after applying all functions.
        torch.Tensor
            Cumulative log absolute determinant across all transformations.
        """
        batch_size = z.shape[0]
        outputs = z
        total_logabsdet = torch.zeros(batch_size)
        for func in funcs:
            if context is not None:
                outputs, logabsdet = func(outputs, context)
            else:
                outputs, logabsdet = func(outputs)
            total_logabsdet += logabsdet
        return outputs, total_logabsdet

    def forward(self, z, context=None):
        """
        Apply forward transformation through all flows in sequence.

        Parameters
        ----------
        z : torch.Tensor
            Input tensor of shape (batch_size, ...).
        context : torch.Tensor, optional
            Context tensor for conditional flows, by default None.

        Returns
        -------
        torch.Tensor
            Transformed tensor.
        torch.Tensor
            Total log absolute determinant.
        """
        funcs = self._flows
        return self._cascade(z, funcs, context)

    def inverse(self, z, context=None):
        """
        Apply inverse transformation through all flows in reverse sequence.

        Parameters
        ----------
        z : torch.Tensor
            Input tensor of shape (batch_size, ...).
        context : torch.Tensor, optional
            Context tensor for conditional flows, by default None.

        Returns
        -------
        torch.Tensor
            Inverse transformed tensor.
        torch.Tensor
            Total log absolute determinant of inverse.
        """
        funcs = (flow.inverse for flow in self._flows[::-1])
        return self._cascade(z, funcs, context)


# Modified from normflows.core.NormalizingFlow
class NormalizingFlow(nn.Module):
    """
    Normalizing Flow model for density estimation and sampling.

    A normalizing flow transforms a simple base distribution through a sequence
    of invertible transformations to model complex target distributions.
    This implementation follows the change of variables formula to compute
    exact likelihoods and enables both density estimation and sampling.

    Attributes
    ----------
    q0 : torch.distributions.Distribution
        Base distribution, typically chosen to be simple (e.g., Standard Normal).
    flows : nn.ModuleList
        Sequence of invertible flow transformations that progressively transform
        the base distribution to the target distribution.
    p : torch.distributions.Distribution or None
        Target distribution to approximate. If provided, enables computation of
        reverse KL divergence for variational inference.
    _transform : Composite
        Composite transformation representing the composition of all flow layers.
        Used for efficient forward and inverse operations.
    """

    def __init__(self, q0, flows, p=None):
        """
        Initialize the Normalizing Flow model.

        Parameters
        ----------
        q0 : torch.distributions.Distribution
            Base distribution. Typically a simple distribution like Standard Normal
            or Uniform that is easy to sample from and evaluate.
        flows : list of nn.Module
            Sequence of invertible flow transformations. Each transformation must
            implement forward() and inverse() methods with log determinant computation.
        p : torch.distributions.Distribution, optional
            Target distribution to approximate. Required for variational inference
            applications where the goal is to minimize KL divergence to p.

        Notes
        -----
        The normalizing flow model defines a distribution q(x) through the change
        of variables formula:
            q(x) = q0(f^{-1}(x)) * |det J_{f^{-1}}(x)|
        where f is the composition of all flow transformations and J_{f^{-1}} is
        the Jacobian of the inverse transformation.
        """
        super().__init__()
        self.q0 = q0
        self.flows = nn.ModuleList(flows)
        self.p = p
        self._transform = Composite(flows)  # Added by NextBayes

    def forward(self, z):
        """Transform latent variables z to data space x.

        Parameters
        ----------
        z : torch.Tensor
            Batch of samples from latent space.

        Returns
        -------
        x : torch.Tensor
            Transformed samples in data space.
        """
        for flow in self.flows:
            z, _ = flow(z)
        return z

    def forward_and_log_det(self, z):
        """Transform z to x and compute log determinant of Jacobian.

        Parameters
        ----------
        z : torch.Tensor
            Batch of samples from latent space.

        Returns
        -------
        x : torch.Tensor
            Transformed samples in data space.
        log_det : torch.Tensor
            Log determinant of the Jacobian matrix.
        """
        log_det = torch.zeros(len(z), device=z.device)
        for flow in self.flows:
            z, log_d = flow(z)
            log_det += log_d
        return z, log_det

    def inverse(self, x):
        """Transform data variables x back to latent space z.

        Parameters
        ----------
        x : torch.Tensor
            Batch of samples from data space.

        Returns
        -------
        z : torch.Tensor
            Transformed samples in latent space.
        """
        for i in range(len(self.flows) - 1, -1, -1):
            x, _ = self.flows[i].inverse(x)
        return x

    def inverse_and_log_det(self, x):
        """Transform x to z and compute log determinant of Jacobian.

        Parameters
        ----------
        x : torch.Tensor
            Batch of samples from data space.

        Returns
        -------
        z : torch.Tensor
            Transformed samples in latent space.
        log_det : torch.Tensor
            Log determinant of the Jacobian matrix.
        """
        log_det = torch.zeros(len(x), device=x.device)
        for i in range(len(self.flows) - 1, -1, -1):
            x, log_d = self.flows[i].inverse(x)
            log_det += log_d
        return x, log_det

    def forward_kld(self, x):
        """Estimate forward KL divergence: KL(p || q).

        Uses samples from the target distribution p. See:
        https://jmlr.org/papers/volume22/19-1028/19-1028.pdf

        Parameters
        ----------
        x : torch.Tensor
            Batch sampled from target distribution p.

        Returns
        -------
        loss : torch.Tensor
            Estimate of forward KL divergence averaged over batch.
        """
        log_q = torch.zeros(len(x), device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z)
        return -torch.mean(log_q)

    def reverse_kld(self, num_samples=1, beta=1.0, score_fn=True):
        """Estimate reverse KL divergence: KL(q || p).

        Uses samples from the base distribution. See:
        https://jmlr.org/papers/volume22/19-1028/19-1028.pdf

        Parameters
        ----------
        num_samples : int, default=1
            Number of samples to draw from base distribution.
        beta : float, default=1.0
            Annealing parameter for tempered distributions.
        score_fn : bool, default=True
            Whether to use score function gradient estimator.

        Returns
        -------
        loss : torch.Tensor
            Estimate of reverse KL divergence averaged over samples.
        """
        z, log_q_ = self.q0(num_samples)
        log_q = torch.zeros_like(log_q_)
        log_q += log_q_
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        if not score_fn:
            z_ = z
            log_q = torch.zeros(len(z_), device=z_.device)  # TODO?
            set_requires_grad(self, False)  # TODO ?
            for i in range(len(self.flows) - 1, -1, -1):
                z_, log_det = self.flows[i].inverse(z_)
                log_q += log_det
            log_q += self.q0.log_prob(z_)
            set_requires_grad(self, True)  # TODO ?
        log_p = self.p.log_prob(z)
        return torch.mean(log_q) - beta * torch.mean(log_p)

    def sample(self, num_samples=1):
        """Generate samples from the flow-based distribution.

        Parameters
        ----------
        num_samples : int, default=1
            Number of samples to generate.

        Returns
        -------
        samples : torch.Tensor
            Generated samples from the model.
        log_prob : torch.Tensor
            Log probabilities of the generated samples.
        """
        z, log_q = self.q0(num_samples)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        return z, log_q

    def log_prob(self, x):
        """Compute log probability of observations under the model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of observations.

        Returns
        -------
        log_prob : torch.Tensor
            Log probabilities for each observation.
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z)
        return log_q

    def save(self, path):
        """Save model state dictionary to file.

        Parameters
        ----------
        path : str
            File path where to save the model state.
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load model state dictionary from file.

        Parameters
        ----------
        path : str
            File path from which to load the model state.
        """
        self.load_state_dict(torch.load(path))


# Modified from normflows.core.ConditionalNormalizingFlow
class ConditionalNormalizingFlow(nn.Module):
    """
    Conditional normalizing flow model.

    Provides condition/context to base distribution and flow layers.

    Parameters
    ----------
    q0 : nn.Module
        Base distribution.
    flows : list of nn.Module
        List of flow layers.
    p : nn.Module, optional
        Target distribution, by default None.
    """

    def __init__(self, q0, flows, p=None):
        super().__init__()
        self.q0 = q0
        self.flows = nn.ModuleList(flows)
        self.p = p
        self._transform = Composite(flows)  # Added by NextBayes

    def forward(self, z, context=None):
        """
        Forward transform: z -> x.

        Parameters
        ----------
        z : torch.Tensor
            Latent variables, shape (N, D_z).
        context : torch.Tensor, optional
            Conditional variables, shape (N, D_context).

        Returns
        -------
        x : torch.Tensor
            Transformed variables, shape (N, D_x).
        """
        for flow in self.flows:
            z, _ = flow(z, context=context)
        return z

    def forward_and_log_det(self, z, context=None):
        """
        Forward transform with log determinant.

        Parameters
        ----------
        z : torch.Tensor
            Latent variables, shape (N, D_z).
        context : torch.Tensor, optional
            Conditional variables, shape (N, D_context).

        Returns
        -------
        x : torch.Tensor
            Transformed variables, shape (N, D_x).
        log_det : torch.Tensor
            Log determinant of Jacobian, shape (N,).
        """
        # shape: (N,)
        log_det = torch.zeros(len(z), device=z.device)

        for flow in self.flows:
            z, log_d = flow(z, context=context)
            log_det += log_d  # accumulate log determinant

        return z, log_det

    def inverse(self, x, context=None):
        """
        Inverse transform: x -> z.

        Parameters
        ----------
        x : torch.Tensor
            Data variables, shape (N, D_x).
        context : torch.Tensor, optional
            Conditional variables, shape (N, D_context).

        Returns
        -------
        z : torch.Tensor
            Latent variables, shape (N, D_z).
        """
        for i in range(len(self.flows) - 1, -1, -1):
            x, _ = self.flows[i].inverse(x, context=context)
        return x

    def inverse_and_log_det(self, x, context=None):
        """
        Inverse transform with log determinant.

        Parameters
        ----------
        x : torch.Tensor
            Data variables, shape (N, D_x).
        context : torch.Tensor, optional
            Conditional variables, shape (N, D_context).

        Returns
        -------
        z : torch.Tensor
            Latent variables, shape (N, D_z).
        log_det : torch.Tensor
            Log determinant of Jacobian, shape (N,).
        """
        # shape: (N,)
        log_det = torch.zeros(len(x), device=x.device)

        for i in range(len(self.flows) - 1, -1, -1):
            x, log_d = self.flows[i].inverse(x, context=context)
            log_det += log_d  # accumulate log determinant

        return x, log_det

    def forward_kld(self, x, context=None):
        """
        Forward KL divergence: KL(p_data || q_model).

        Parameters
        ----------
        x : torch.Tensor
            Data from target distribution, shape (N, D_x).
        context : torch.Tensor, optional
            Conditional variables, shape (N, D_context).

        Returns
        -------
        kl_div : torch.Tensor
            KL divergence estimate (scalar).
        """
        # Compute log probability under model
        log_q = torch.zeros(len(x), device=x.device)
        z = x

        # Transform to latent space
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z, context=context)
            log_q += log_det

        # Add base distribution
        log_q += self.q0.log_prob(z, context=context)

        # KL(p||q) = -E_p[log q]
        return -torch.mean(log_q)

    def reverse_kld(self, num_samples=1, beta=1.0, score_fn=True, context=None):
        """
        Reverse KL divergence: KL(q_model || p_target).

        Parameters
        ----------
        num_samples : int
            Number of samples.
        beta : float
            Annealing parameter.
        score_fn : bool
            Whether to use score function gradient.
        context : torch.Tensor, optional
            Conditional variables, shape (num_samples, D_context).

        Returns
        -------
        kl_div : torch.Tensor
            KL divergence estimate (scalar).
        """
        # Sample from base distribution
        z, log_q_ = self.q0(num_samples, context=context)
        log_q = torch.zeros_like(log_q_)
        log_q += log_q_

        # Forward transform
        for flow in self.flows:
            z, log_det = flow(z, context=context)
            log_q -= log_det

        # Recompute without gradient if not using score function
        if not score_fn:
            z_ = z
            log_q = torch.zeros(len(z_), device=z_.device)
            set_requires_grad(self, False)

            for i in range(len(self.flows) - 1, -1, -1):
                z_, log_det = self.flows[i].inverse(z_, context=context)
                log_q += log_det

            log_q += self.q0.log_prob(z_, context=context)
            set_requires_grad(self, True)

        # Target distribution log probability
        log_p = self.p.log_prob(z, context=context)

        # KL(q||p) = E_q[log q - log p]
        return torch.mean(log_q) - beta * torch.mean(log_p)

    def sample(self, num_samples=1, context=None):
        """
        Generate samples from the model.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        context : torch.Tensor, optional
            Conditional variables, shape (num_samples, D_context).

        Returns
        -------
        samples : torch.Tensor
            Generated samples, shape (num_samples, D_x).
        log_prob : torch.Tensor
            Log probabilities, shape (num_samples,).
        """
        # Sample from base distribution
        z, log_q = self.q0(num_samples, context=context)

        # Transform through flows
        for flow in self.flows:
            z, log_det = flow(z, context=context)
            log_q -= log_det

        return z, log_q

    def log_prob(self, x, context=None):
        """
        Compute log probability of data.

        Parameters
        ----------
        x : torch.Tensor
            Data points, shape (N, D_x).
        context : torch.Tensor, optional
            Conditional variables, shape (N, D_context).

        Returns
        -------
        log_prob : torch.Tensor
            Log probabilities, shape (N,).
        """
        # Initialize log probability
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x

        # Inverse transform to latent space
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z, context=context)
            log_q += log_det  # accumulate log determinant

        # Add base distribution log probability
        log_q += self.q0.log_prob(z, context=context)

        return log_q

    def save(self, path):
        """
        Save model state dictionary to file.

        Parameters
        ----------
        path : str
            File path where to save the model state.
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """
        Load model state dictionary from file.

        Parameters
        ----------
        path : str
            File path from which to load the model state.
        """
        self.load_state_dict(torch.load(path))
