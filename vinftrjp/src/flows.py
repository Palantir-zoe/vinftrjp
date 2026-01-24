import nflows.utils.typechecks as check
import numpy as np
import torch
from nflows.distributions.normal import StandardNormal
from scipy.special import logsumexp
from scipy.stats import Uniform
from torch import nn, optim

from src.transforms import (
    CompositeCDFTransform,
    CompositeTransform,
    IdentityTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
    ReversePermutation,
)
from src.utils import torchutils


class Distribution(nn.Module):
    """
    Base class for all probability distribution objects.

    Provides interface for log probability evaluation and sampling.
    """

    def forward(self, *args):
        """Forward method cannot be called for Distribution objects."""
        raise RuntimeError("Forward method cannot be called for a Distribution object.")

    def log_prob(self, inputs, context=None):
        """
        Calculate log probability under the distribution.

        Parameters
        ----------
        inputs : torch.Tensor
            Input variables of shape [batch_size, ...]
        context : torch.Tensor or None, optional
            Conditioning variables of shape [batch_size, ...]. If None, context is ignored.

        Returns
        -------
        torch.Tensor
            Log probabilities of shape [batch_size]

        Raises
        ------
        ValueError
            If inputs and context have different batch sizes
        """
        inputs = torch.as_tensor(inputs)
        if context is not None:
            context = torch.as_tensor(context)
            if inputs.shape[0] != context.shape[0]:
                raise ValueError("Number of input items must be equal to number of context items.")
        return self._log_prob(inputs, context)

    def _log_prob(self, inputs, context):
        """Subclass must implement log probability calculation."""
        raise NotImplementedError

    def sample(self, num_samples, context=None, batch_size=None):
        """
        Generate samples from the distribution.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate
        context : torch.Tensor or None, optional
            Conditioning variables
        batch_size : int or None, optional
            Number of samples per batch. If None, all samples generated in one batch.

        Returns
        -------
        torch.Tensor
            Samples tensor. Shape [num_samples, ...] if context is None,
            or [context_size, num_samples, ...] if context is given.

        Raises
        ------
        TypeError
            If num_samples or batch_size are not positive integers
        """
        if not check.is_positive_int(num_samples):
            raise TypeError("Number of samples must be a positive integer.")

        if context is not None:
            context = torch.as_tensor(context)

        if batch_size is None:
            return self._sample(num_samples, context)

        else:
            if not check.is_positive_int(batch_size):
                raise TypeError("Batch size must be a positive integer.")

            num_batches = num_samples // batch_size
            num_leftover = num_samples % batch_size
            samples = [self._sample(batch_size, context) for _ in range(num_batches)]
            if num_leftover > 0:
                samples.append(self._sample(num_leftover, context))
            return torch.cat(samples, dim=0)

    def _sample(self, num_samples, context):
        """Subclass must implement sampling procedure."""
        raise NotImplementedError

    def sample_and_log_prob(self, num_samples, context=None):
        """
        Generate samples with their log probabilities.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate
        context : torch.Tensor or None, optional
            Conditioning variables

        Returns
        -------
        tuple
            (samples, log_probs) where:
            - samples: torch.Tensor of shape [num_samples, ...] or [context_size, num_samples, ...]
            - log_probs: torch.Tensor of shape [num_samples] or [context_size, num_samples]
        """
        samples = self.sample(num_samples, context=context)

        if context is not None:
            # Merge context dimension with sample dimension for log_prob calculation
            samples = torchutils.merge_leading_dims(samples, num_dims=2)
            context = torchutils.repeat_rows(context, num_reps=num_samples)
            assert samples.shape[0] == context.shape[0]

        log_prob = self.log_prob(samples, context=context)

        if context is not None:
            # Split context dimension from sample dimension
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])
            log_prob = torchutils.split_leading_dim(log_prob, shape=[-1, num_samples])

        return samples, log_prob

    def mean(self, context=None):
        """Compute mean of the distribution."""
        if context is not None:
            context = torch.as_tensor(context)
        return self._mean(context)

    def _mean(self, context):
        """Subclass must implement mean calculation."""
        raise NotImplementedError


class Flow(Distribution):
    """
    Base class for normalizing flow models.

    A normalizing flow transforms a simple base distribution through
    a sequence of invertible transformations to model complex distributions.
    """

    def __init__(self, transform, distribution, embedding_net=None):
        """
        Initialize flow model.

        Parameters
        ----------
        transform : Transform
            Invertible transformation that maps data to noise
        distribution : Distribution
            Base distribution that generates the noise
        embedding_net : nn.Module or None, optional
            Neural network to encode context variables. If None, identity mapping is used.
        """
        super().__init__()
        self._transform = transform
        self._distribution = distribution
        if embedding_net is not None:
            assert isinstance(embedding_net, torch.nn.Module), (
                "embedding_net is not a nn.Module. "
                "If you want to use hard-coded summary features, "
                "please simply pass the encoded features and pass "
                "embedding_net=None"
            )
            self._embedding_net = embedding_net
        else:
            self._embedding_net = torch.nn.Identity()

    def _log_prob(self, inputs, context):
        """Compute log probability using change of variables formula."""
        embedded_context = self._embedding_net(context)
        noise, logabsdet = self._transform(inputs, context=embedded_context)
        log_prob = self._distribution.log_prob(noise, context=embedded_context)
        return log_prob + logabsdet

    def _sample(self, num_samples, context):
        """Generate samples by transforming base distribution samples."""
        embedded_context = self._embedding_net(context)
        noise = self._distribution.sample(num_samples, context=embedded_context)

        if embedded_context is not None:
            # Merge context dimension with sample dimension for transform application
            noise = torchutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = torchutils.repeat_rows(embedded_context, num_reps=num_samples)

        samples, _ = self._transform.inverse(noise, context=embedded_context)

        if embedded_context is not None:
            # Split context dimension from sample dimension
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])

        return samples

    def sample_and_log_prob(self, num_samples, context=None):
        """
        Efficiently generate samples with log probabilities.

        More efficient than calling sample() and log_prob() separately.
        """
        embedded_context = self._embedding_net(context)
        noise, log_prob = self._distribution.sample_and_log_prob(num_samples, context=embedded_context)

        if embedded_context is not None:
            # Merge context dimension with sample dimension for transform application
            noise = torchutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = torchutils.repeat_rows(embedded_context, num_reps=num_samples)

        samples, logabsdet = self._transform.inverse(noise, context=embedded_context)

        if embedded_context is not None:
            # Split context dimension from sample dimension
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])
            logabsdet = torchutils.split_leading_dim(logabsdet, shape=[-1, num_samples])

        return samples, log_prob - logabsdet

    def transform_to_noise(self, inputs, context=None):
        """
        Transform data to noise space for goodness-of-fit checking.

        Parameters
        ----------
        inputs : torch.Tensor
            Data to transform, shape [batch_size, ...]
        context : torch.Tensor or None, optional
            Conditioning variables

        Returns
        -------
        torch.Tensor
            Noise representation of inputs, shape [batch_size, ...]
        """
        noise, _ = self._transform(inputs, context=self._embedding_net(context))
        return noise


class RationalQuadraticFlowFAV(Flow):
    """
    Rational Quadratic Flow with forward amortized variational inference.

    Implements a normalizing flow with piecewise rational quadratic transforms
    and specialized training procedure.
    """

    @classmethod
    def factory(
        cls,
        inputs,
        base_dist=None,
        boxing_transform=IdentityTransform(),
        initial_transform=IdentityTransform(),
        input_weights=None,
    ):
        """
        Factory method to create and train Rational Quadratic Flow.

        Parameters
        ----------
        inputs : array-like
            Training data of shape (n_samples, n_features)
        base_dist : Distribution or None, optional
            Base distribution. If None, uniform distribution on [0,1]^d is used.
        boxing_transform : Transform, optional
            Transform to map data to base distribution support
        initial_transform : Transform, optional
            Initial transformation applied before flow
        input_weights : array-like or None, optional
            Sample weights for weighted training

        Returns
        -------
        RationalQuadraticFlowFAV
            Trained flow model
        """
        ndim = inputs.shape[1]

        # Configuration
        dim_multiplier = int(np.log2(1 + np.log2(ndim)))
        num_layers = 1 + dim_multiplier  # int(np.log2(ndim))
        num_layers = 3  # fix for now
        num_iter = 3000  # * max(1,dim_multiplier)
        ss_size = 32

        if base_dist is None:
            base_dist = Uniform(low=torch.zeros(ndim), high=torch.ones(ndim))
            utr = IdentityTransform()
            ittr = IdentityTransform()
        else:
            utr = boxing_transform
            ittr = initial_transform

        val_percent = 0.1

        def get_train_val_idx(N, val_percent):
            """Split data into training and validation indices."""
            sn = N * val_percent
            interval = N / sn
            val_idx = np.array(np.arange(sn) * interval + interval / 2, dtype=int)
            train_idx = np.setdiff1d(np.arange(N), val_idx)
            return train_idx, val_idx

        N = inputs.shape[0]
        if input_weights is not None:
            train_idx, validate_idx = get_train_val_idx(N, val_percent)
            weight_sort_idx = np.argsort(input_weights)
            train_idx = weight_sort_idx[train_idx]
            validate_idx = weight_sort_idx[validate_idx]

            x_train = torch.tensor(inputs[train_idx], dtype=torch.float32)
            x_train_w = input_weights[train_idx]
            x_train_w = np.exp(np.log(x_train_w) - logsumexp(np.log(x_train_w)))
            x_validate = torch.tensor(inputs[validate_idx], dtype=torch.float32)
            x_validate_w = input_weights[validate_idx]
            x_validate_w = np.exp(np.log(x_validate_w) - logsumexp(np.log(x_validate_w)))
        else:
            train_idx, validate_idx = get_train_val_idx(N, val_percent)
            x_train = torch.tensor(inputs[train_idx], dtype=torch.float32)
            x_train_w = np.full(train_idx.shape[0], 1.0 / train_idx.shape[0])
            x_validate = torch.tensor(inputs[validate_idx], dtype=torch.float32)
            x_validate_w = np.full(validate_idx.shape[0], 1.0 / validate_idx.shape[0])

        transforms = []

        for _ in range(num_layers):
            transforms.append(ReversePermutation(features=ndim))
            transforms.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=ndim, hidden_features=ndim * 32, num_blocks=2, num_bins=10
                )
            )

        _transform = CompositeTransform(
            [
                ittr,
                CompositeCDFTransform(utr, CompositeTransform(transforms)),
            ]
        )
        myflow = cls(_transform, base_dist)
        optimizer = optim.Adam(myflow.parameters(), lr=3e-3)
        lmbda = lambda epoch: 0.995 if epoch < 200 else 0.9971
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
        train_hloss = torch.zeros(num_iter)
        val_hloss = torch.zeros(num_iter)
        train_lastloss = float("inf")
        val_lastloss = float("inf")
        if True:
            for i in range(num_iter):
                optimizer.zero_grad()
                ids = np.random.choice(x_train.shape[0], ss_size, p=x_train_w)
                loss = -myflow.log_prob(inputs=x_train[ids]).mean()
                loss.backward()
                optimizer.step()
                scheduler.step()
                with torch.no_grad():
                    # do validation here
                    val_ids = np.random.choice(x_validate.shape[0], ss_size, p=x_validate_w)
                    val_loss = -myflow.log_prob(inputs=x_validate[val_ids]).mean()
                    val_hloss[i] = val_loss  # store for assessment
                    train_hloss[i] = loss  # store for assessment
                    train_avgloss = train_hloss[max(0, i - 99) : i + 1].mean()
                    val_avgloss = val_hloss[max(0, i - 99) : i + 1].mean()
                    if (i) % 100 == 0:
                        print(i, train_avgloss, val_avgloss)
                        # if i>0 and val_lastloss - val_avgloss < 0.05:
                        if i > 0 and val_lastloss - val_avgloss < 0:
                            print("Finished training")
                            print(i, train_avgloss, val_avgloss)
                            break
                        elif i > 0:
                            val_lastloss = val_avgloss
        return myflow


class RationalQuadraticFlow2(Flow):
    """
    Alternative implementation of Rational Quadratic Flow.

    Simplified training procedure without validation split.
    """

    @classmethod
    def factory(
        cls,
        inputs,
        base_dist=None,
        boxing_transform=IdentityTransform(),
        initial_transform=IdentityTransform(),
        input_weights=None,
    ):
        """
        Factory method to create and train Rational Quadratic Flow.

        Parameters
        ----------
        inputs : array-like
            Training data of shape (n_samples, n_features)
        base_dist : Distribution or None, optional
            Base distribution. If None, uniform distribution on [0,1]^d is used.
        boxing_transform : Transform, optional
            Transform to map data to base distribution support
        initial_transform : Transform, optional
            Initial transformation applied before flow
        input_weights : array-like or None, optional
            Sample weights for weighted training

        Returns
        -------
        RationalQuadraticFlow2
            Trained flow model
        """
        ndim = inputs.shape[1]

        # Configuration
        dim_multiplier = int(np.log2(1 + np.log2(ndim)))
        num_layers = 1 + dim_multiplier  # int(np.log2(ndim))
        num_layers = 3  # fix for now
        num_iter = 1000  # * max(1,dim_multiplier)
        ss_size = 128

        if base_dist is None:
            base_dist = Uniform(low=torch.zeros(ndim), high=torch.ones(ndim))
            utr = IdentityTransform()
            ittr = IdentityTransform()
        else:
            utr = boxing_transform
            ittr = initial_transform

        x = torch.tensor(inputs, dtype=torch.float32)

        transforms = []

        for _ in range(num_layers):
            transforms.append(ReversePermutation(features=ndim))
            transforms.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=ndim, hidden_features=ndim * 32, num_blocks=2, num_bins=10
                )
            )

        _transform = CompositeTransform(
            [
                ittr,
                CompositeCDFTransform(utr, CompositeTransform(transforms)),
            ]
        )
        xx, __ = _transform.forward(x)
        myflow = cls(_transform, base_dist)
        optimizer = optim.Adam(myflow.parameters(), lr=3e-3)
        lmbda = lambda epoch: 0.992 if epoch < 200 else 0.9971
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
        hloss = torch.zeros(num_iter)
        if input_weights is None:
            for i in range(num_iter):
                optimizer.zero_grad()
                ids = np.random.choice(x.shape[0], ss_size)
                loss = -myflow.log_prob(inputs=x[ids]).mean()
                with torch.no_grad():
                    hloss[i] = loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                if (i) % 100 == 0:
                    print(i, hloss[max(0, i - 50) : i + 1].mean())
                if (i) % 250 == 0 and (i) > 0:
                    ss_size *= 2
        else:
            for i in range(num_iter):
                optimizer.zero_grad()
                ids = np.random.choice(x.shape[0], ss_size, p=input_weights)
                loss = -myflow.log_prob(inputs=x[ids]).mean()
                with torch.no_grad():
                    hloss[i] = loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                if (i) % 100 == 0:
                    print(i, hloss[max(0, i - 50) : i + 1].mean())
                if (i) % 250 == 0 and (i) > 0:
                    ss_size *= 2
        return myflow


class ConditionalMaskedRationalQuadraticFlow(Flow):
    """
    Conditional Rational Quadratic Flow with masking for missing data.

    Handles conditional generation and inference with partially observed data.
    """

    @classmethod
    def factory(
        cls,
        inputs,
        context_inputs,
        context_mask,
        aux_dist=StandardNormal((1,)),
        base_dist=None,
        boxing_transform=IdentityTransform(),
        initial_transform=IdentityTransform(),
        input_weights=None,
    ):
        """
        Factory method to create and train Conditional Masked Rational Quadratic Flow.

        Parameters
        ----------
        inputs : array-like
            Training data of shape (n_samples, n_features)
        context_inputs : array-like
            Context variables of shape (n_samples, n_context_features)
        context_mask : array-like
            Boolean mask indicating observed/missing data
        aux_dist : Distribution, optional
            Auxiliary distribution for imputing missing values
        base_dist : Distribution or None, optional
            Base distribution. If None, uniform distribution on [0,1]^d is used.
        boxing_transform : Transform, optional
            Transform to map data to base distribution support
        initial_transform : Transform, optional
            Initial transformation applied before flow
        input_weights : array-like or None, optional
            Sample weights for weighted training

        Returns
        -------
        ConditionalMaskedRationalQuadraticFlow
            Trained conditional flow model
        """
        ndim = inputs.shape[1]
        ncdim = context_inputs.shape[1]

        # Configuration
        dim_multiplier = int(np.log2(1 + np.log2(ndim)))
        num_layers = 1 + dim_multiplier  # int(np.log2(ndim))
        num_layers = 3  # fix for now
        num_iter = 1000  # * max(1,dim_multiplier)
        ss_size = 128

        if base_dist is None:
            base_dist = Uniform(low=torch.zeros(ndim), high=torch.ones(ndim))
            utr = IdentityTransform()
            ittr = IdentityTransform()
        else:
            utr = boxing_transform
            ittr = initial_transform

        x = torch.tensor(inputs, dtype=torch.float32)
        y = torch.tensor(context_inputs, dtype=torch.float32)

        transforms = []

        for _ in range(num_layers):
            transforms.append(ReversePermutation(features=ndim))
            transforms.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=ndim,
                    hidden_features=ndim * 32,
                    context_features=ncdim,
                    num_blocks=2,
                    num_bins=10,
                )
            )

        _transform = CompositeTransform(
            [
                ittr,
                CompositeCDFTransform(utr, CompositeTransform(transforms)),
            ]
        )
        myflow = cls(_transform, base_dist)
        optimizer = optim.Adam(myflow.parameters(), lr=3e-3)
        lmbda = lambda epoch: 0.992 if epoch < 200 else 0.9971
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
        hloss = torch.zeros(num_iter)
        lastloss = 1e9
        if input_weights is None:
            for i in range(num_iter):
                optimizer.zero_grad()
                ids = np.random.choice(x.shape[0], ss_size)
                x_m = torch.zeros_like(x[ids])
                x_m[~context_mask[ids]] = x[ids][~context_mask[ids]]
                x_m[context_mask[ids]] = aux_dist.sample((int(context_mask[ids].sum()),)).flatten()
                loss = -myflow.log_prob(inputs=x_m, context=y[ids]).mean()
                with torch.no_grad():
                    hloss[i] = loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                if (i) % 100 == 0:
                    avgloss = hloss[max(0, i - 50) : i + 1].mean()
                    print(i, avgloss)
                    if lastloss - avgloss < 0.01:
                        break
                    lastloss = avgloss
                if (i) % 200 == 0 and (i) > 0:
                    ss_size *= 2
        else:
            for i in range(num_iter):
                optimizer.zero_grad()
                ids = np.random.choice(x.shape[0], ss_size, p=input_weights)
                x_m = torch.zeros_like(x[ids])
                x_m[~context_mask[ids]] = x[ids][~context_mask[ids]]
                x_m[context_mask[ids]] = aux_dist.sample((int(context_mask[ids].sum()),)).flatten()
                loss = -myflow.log_prob(inputs=x_m, context=y[ids]).mean()
                with torch.no_grad():
                    hloss[i] = loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                if (i) % 100 == 0:
                    avgloss = hloss[max(0, i - 50) : i + 1].mean()
                    print(i, avgloss)
                    if lastloss - avgloss < 0.01:
                        break
                    lastloss = avgloss
                if (i) % 200 == 0 and (i) > 0:
                    ss_size *= 2
        return myflow
