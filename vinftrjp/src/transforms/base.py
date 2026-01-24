import sys

import nflows.utils.typechecks as check
import numpy as np
import torch
from nflows.transforms import made as made_module
from nflows.transforms.splines import rational_quadratic
from nflows.transforms.splines.rational_quadratic import (
    rational_quadratic_spline,
    unconstrained_rational_quadratic_spline,
)
from torch import Tensor, nn
from torch.nn import functional as F

from src.utils import torchutils
from src.utils.linalgtools import make_pos_def, safe_cholesky


class InputOutsideDomain(Exception):
    """Exception to be thrown when the input to a transform is not within its domain."""

    pass


class InverseNotAvailable(Exception):
    """Exception to be thrown when a transform does not have an inverse."""

    pass


class Transform(nn.Module):
    """Base class for all transform objects."""

    def forward(self, inputs, context=None):
        raise NotImplementedError()

    def inverse(self, inputs, context=None):
        raise InverseNotAvailable()


class Permutation(Transform):
    """Permutes inputs on a given dimension using a given permutation."""

    def __init__(self, permutation, dim=1):
        if permutation.ndimension() != 1:
            raise ValueError("Permutation must be a 1D tensor.")
        if not check.is_positive_int(dim):
            raise ValueError("dim must be a positive integer.")

        super().__init__()
        self._dim = dim
        self.register_buffer("_permutation", permutation)

    @property
    def _inverse_permutation(self):
        return torch.argsort(self._permutation)

    @staticmethod
    def _permute(inputs, permutation, dim):
        if dim >= inputs.ndimension():
            raise ValueError("No dimension {} in inputs.".format(dim))
        if inputs.shape[dim] != len(permutation):
            raise ValueError("Dimension {} in inputs must be of size {}.".format(dim, len(permutation)))
        batch_size = inputs.shape[0]
        outputs = torch.index_select(inputs, dim, permutation)
        logabsdet = inputs.new_zeros(batch_size)
        return outputs, logabsdet

    def forward(self, inputs, context=None):
        return self._permute(inputs, self._permutation, self._dim)

    def inverse(self, inputs, context=None):
        return self._permute(inputs, self._inverse_permutation, self._dim)


class ReversePermutation(Permutation):
    """Reverses the elements of the input. Only works with 1D inputs."""

    def __init__(self, features, dim=1):
        if not check.is_positive_int(features):
            raise ValueError("Number of features must be a positive integer.")
        super().__init__(torch.arange(features - 1, -1, -1), dim)


class LogTransform(Transform):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, inputs, context=None):
        if torch.min(inputs) <= 0:
            print("Inputs negative ", inputs[inputs <= 0].shape)
            raise InputOutsideDomain()
        inputs = torch.clamp(inputs, self.eps, None)

        outputs = torch.log(inputs)
        if inputs.ndimension() > 1:
            sumdims = list(range(1, inputs.ndimension()))
            logabsdet = torchutils.sum_except_batch(-torch.log(inputs), num_batch_dims=1)
        else:
            logabsdet = -torch.log(inputs)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        outputs = torch.exp(inputs)
        if inputs.ndimension() > 1:
            sumdims = list(range(1, inputs.ndimension()))
            logabsdet = -torchutils.sum_except_batch(-torch.log(outputs), num_batch_dims=1)
        else:
            logabsdet = torch.log(outputs)
        return outputs, logabsdet


class ColumnSpecificTransform(Transform):
    def __init__(self, spec={}):
        super().__init__()
        for col, t in spec.items():
            assert isinstance(t, Transform)
            assert isinstance(col, int)
        self._spec = spec

    def forward(self, inputs, context=None):
        outputs = inputs.clone()
        ld = torch.zeros(inputs.shape[0])
        for col, t in self._spec.items():
            v, ldtemp = t.forward(inputs[:, col].reshape((inputs.shape[0], 1)), context)
            outputs[:, col] = v.reshape((inputs.shape[0],))
            ld += ldtemp
        return outputs, ld

    def inverse(self, inputs, context=None):
        outputs = inputs.clone()
        ld = torch.zeros(inputs.shape[0])
        for col, t in self._spec.items():
            v, ldtemp = t.inverse(inputs[:, col].reshape((inputs.shape[0], 1)), context)
            outputs[:, col] = v.reshape((inputs.shape[0],))
            ld += ldtemp
        return outputs, ld


class CompositeTransform(Transform):
    """Composes several transforms into one, in the order they are given."""

    def __init__(self, transforms):
        """Constructor.

        Args:
            transforms: an iterable of `Transform` objects.
        """
        super().__init__()
        self._transforms = nn.ModuleList(transforms)

    @staticmethod
    def _cascade(inputs, funcs, context):
        batch_size = inputs.shape[0]
        outputs = inputs
        total_logabsdet = inputs.new_zeros(batch_size)
        for func in funcs:
            outputs, logabsdet = func(outputs, context)
            total_logabsdet += logabsdet
        return outputs, total_logabsdet

    def forward(self, inputs, context=None):
        funcs = self._transforms
        return self._cascade(inputs, funcs, context)

    def inverse(self, inputs, context=None):
        funcs = (transform.inverse for transform in self._transforms[::-1])
        return self._cascade(inputs, funcs, context)


class InverseTransform(Transform):
    """Creates a transform that is the inverse of a given transform."""

    def __init__(self, transform):
        """Constructor.

        Args:
            transform: An object of type `Transform`.
        """
        super().__init__()
        self._transform = transform

    def forward(self, inputs, context=None):
        return self._transform.inverse(inputs, context)

    def inverse(self, inputs, context=None):
        return self._transform(inputs, context)


class Sigmoid(Transform):
    def __init__(self, temperature=1, eps=1e-6, learn_temperature=False):
        super().__init__()
        self.eps = eps
        if learn_temperature:
            self.temperature = nn.Parameter(torch.Tensor([temperature]))
        else:
            self.temperature = torch.Tensor([temperature])

    def forward(self, inputs, context=None):
        inputs = self.temperature * inputs
        outputs = torch.sigmoid(inputs)
        logabsdet = torchutils.sum_except_batch(torch.log(self.temperature) - F.softplus(-inputs) - F.softplus(inputs))
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if torch.min(inputs) < 0 or torch.max(inputs) > 1:
            raise InputOutsideDomain()

        inputs = torch.clamp(inputs, self.eps, 1 - self.eps)

        outputs = (1 / self.temperature) * (torch.log(inputs) - torch.log1p(-inputs))
        logabsdet = -torchutils.sum_except_batch(
            torch.log(self.temperature)
            - F.softplus(-self.temperature * outputs)
            - F.softplus(self.temperature * outputs)
        )
        return outputs, logabsdet


class IdentityTransform(Transform):
    """Transform that leaves input unchanged."""

    def forward(self, inputs: Tensor, context=Tensor | None):
        batch_size = inputs.size(0)
        logabsdet = inputs.new_zeros(batch_size)
        return inputs, logabsdet

    def inverse(self, inputs: Tensor, context=Tensor | None):
        return self(inputs, context)


class CauchyCDF(Transform):
    def __init__(self, location=None, scale=None, features=None):
        super().__init__()

    def forward(self, inputs, context=None):
        outputs = (1 / np.pi) * torch.atan(inputs) + 0.5
        logabsdet = torchutils.sum_except_batch(-np.log(np.pi) - torch.log(1 + inputs**2))
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if torch.min(inputs) < 0 or torch.max(inputs) > 1:
            raise InputOutsideDomain()

        outputs = torch.tan(np.pi * (inputs - 0.5))
        logabsdet = -torchutils.sum_except_batch(-np.log(np.pi) - torch.log(1 + outputs**2))
        return outputs, logabsdet


class FixedNorm(Transform):
    def __init__(self, inputs, weights=None):
        super().__init__()
        self._dim = inputs.shape[1]
        self._scale = torch.ones(self._dim)
        self._shift = torch.zeros(self._dim)
        if weights is None:
            n = inputs.shape[0]
            weights = torch.full([n], 1.0 / n)
        for d in range(self._dim):
            self._shift[d] = (inputs[:, d] @ weights) / weights.sum()  # torch.mean(inputs[:,d])
            cov = torch.Tensor(
                np.cov(
                    inputs[:, d].detach().numpy().flatten(),
                    aweights=weights.detach().numpy(),
                )
            )  # workaround for torch 1.9.1
            self._scale[d] = cov**0.5
            self._scale[d] = max(self._scale[d], 1e-10)
            if torch.any(~torch.isfinite(self._scale)):
                print("inputs are singular, ", d)
                print(inputs[~torch.isfinite(inputs[:, d])])
                print(self._scale)
                print(inputs)
                print(cov)
                print(weights)
                sys.exit(0)

    def inverse(self, inputs, context=None):
        outputs = self._scale * inputs + self._shift
        return outputs, torch.log(torch.abs(self._scale * torch.ones_like(inputs))).sum(axis=-1)

    def forward(self, inputs, context=None):
        outputs = 1.0 / self._scale * (inputs - self._shift)
        return outputs, -torch.log(torch.abs(self._scale * torch.ones_like(inputs))).sum(axis=-1)


class LTransform(Transform):
    def __init__(self, M=None, dim=None):
        super().__init__()
        if M is None:
            assert dim is not None
            self.L = torch.eye(dim)
        else:
            self.L = M
            self.Linv = torch.linalg.inv(M)
            self.ld = torch.logdet(M)

    def forward(self, X, context=None):
        return torch.matmul(X, self.L.T), torch.full([X.shape[0]], self.ld)

    def inverse(self, X, context=None):
        return torch.matmul(X, self.Linv.T), torch.full([X.shape[0]], -self.ld)


class L1DTransform(Transform):
    def __init__(self, M=None):
        super().__init__()
        if M is None:
            self.L = 1
            self.ld = 1
        else:
            self.L = M
            self.Linv = 1.0 / M
            self.ld = np.log(M)

    def forward(self, X, context=None):
        return X * self.L, torch.full([X.shape[0]], self.ld)

    def inverse(self, X, context=None):
        return X * self.Linv, torch.full([X.shape[0]], -self.ld)


class SinArcSinhTransform(Transform):
    def __init__(self, e, d):
        super().__init__()
        assert isinstance(e, int) or isinstance(e, float)
        assert isinstance(d, int) or isinstance(d, float)
        self.epsilon = e
        self.delta = d

    def _sas(self, x, epsilon, delta):
        return torch.sinh((torch.arcsinh(x) + epsilon) / delta)

    def _isas(self, x, epsilon, delta):
        return torch.sinh(delta * torch.arcsinh(x) - epsilon)

    def _ldisas(self, x, epsilon, delta):
        return torch.log(torch.abs(delta * torch.cosh(epsilon - delta * torch.arcsinh(x)) / torch.sqrt(1 + x**2)))

    def forward(self, X, context=None):
        XX = self._sas(X, self.epsilon, self.delta)
        ld = -self._ldisas(XX, self.epsilon, self.delta)
        ld = ld.flatten()
        return XX, ld

    def inverse(self, X, context=None):
        ld = self._ldisas(X, self.epsilon, self.delta)
        ld = ld.flatten()
        return self._isas(X, self.epsilon, self.delta), ld


class SAS2DTransform(Transform):
    def __init__(self, e=[0, 0], d=[1, 1]):
        super().__init__()
        self.epsilon = e
        self.delta = d
        self.t1 = SinArcSinhTransform(e[0], d[0])
        self.t2 = SinArcSinhTransform(e[1], d[1])

    def forward(self, X, context=None):
        TX = torch.zeros_like(X)
        ld = torch.zeros_like(X)
        TX[:, 0], ld[:, 0] = self.t1.forward(X[:, 0])
        TX[:, 1], ld[:, 1] = self.t2.forward(X[:, 1])
        return TX, ld.sum(axis=-1)

    def inverse(self, X, context=None):
        TX = torch.zeros_like(X)
        ld = torch.zeros_like(X)
        TX[:, 0], ld[:, 0] = self.t1.inverse(X[:, 0])
        TX[:, 1], ld[:, 1] = self.t2.inverse(X[:, 1])
        return TX, ld.sum(axis=-1)


class NaiveGaussianTransform(Transform):
    def __init__(self, inputs, weights=None):
        super().__init__()
        self._dim = inputs.shape[1]
        self._shift = torch.zeros(self._dim)
        if weights == None:
            n = inputs.shape[0]
            weights = torch.full([n], 1.0 / n)
        for d in range(self._dim):
            self._shift[d] = (inputs[:, d] @ weights) / weights.sum()  # torch.mean(inputs[:,d])
        # fit covariance to inputs, then decompose
        if self._dim == 1:
            # 1D
            # std = torch.std(inputs,unbiased=True)
            cov = torch.Tensor(
                np.cov(inputs.detach().numpy().flatten(), aweights=weights.detach().numpy())
            )  # workaround for torch 1.9.1
            std = cov**0.5
            self._t = L1DTransform(std)
        else:
            # >1D
            # cov = torch.cov(inputs) # not supported in torch 1.9.1
            if inputs.shape[0] <= 1:
                print("WARNING: cov cannot be taken on a single sample.", inputs)
                L = torch.eye(self._dim)
                self._t = LTransform(L)
            else:
                cov = torch.Tensor(
                    np.cov(
                        inputs.detach().numpy(),
                        aweights=weights.detach().numpy(),
                        rowvar=False,
                    )
                )  # workaround for torch 1.9.1
                cov = torch.Tensor(make_pos_def(cov.detach().numpy()))
                L = safe_cholesky(cov)
                self._t = LTransform(L)

    def forward(self, inputs, context=None):
        x, ld = self._t.forward(inputs, context)
        x += self._shift
        return x, ld

    def inverse(self, inputs, context=None):
        x = inputs - self._shift
        return self._t.inverse(x, context)


class AutoregressiveTransform(Transform):
    """Transforms each input variable with an invertible elementwise transformation.

    The parameters of each invertible elementwise transformation can be functions of previous input
    variables, but they must not depend on the current or any following input variables.

    NOTE: Calculating the inverse transform is D times slower than calculating the
    forward transform, where D is the dimensionality of the input to the transform.
    """

    def __init__(self, autoregressive_net):
        super(AutoregressiveTransform, self).__init__()
        self.autoregressive_net = autoregressive_net

    def forward(self, inputs, context=None):
        autoregressive_params = self.autoregressive_net(inputs, context)
        outputs, logabsdet = self._elementwise_forward(inputs, autoregressive_params)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        num_inputs = np.prod(inputs.shape[1:])
        outputs = torch.zeros_like(inputs)
        logabsdet = None
        for _ in range(num_inputs):
            autoregressive_params = self.autoregressive_net(outputs, context)
            outputs, logabsdet = self._elementwise_inverse(inputs, autoregressive_params)
        return outputs, logabsdet

    def _output_dim_multiplier(self):
        raise NotImplementedError()

    def _elementwise_forward(self, inputs, autoregressive_params):
        raise NotImplementedError()

    def _elementwise_inverse(self, inputs, autoregressive_params):
        raise NotImplementedError()


class MaskedPiecewiseRationalQuadraticAutoregressiveTransform(AutoregressiveTransform):
    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        num_blocks=2,
        use_residual_blocks=True,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        min_bin_width=rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=rational_quadratic.DEFAULT_MIN_DERIVATIVE,
    ):
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = tails
        self.tail_bound = tail_bound

        autoregressive_net = made_module.MADE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )

        super().__init__(autoregressive_net)

    def _output_dim_multiplier(self):
        if self.tails == "linear":
            return self.num_bins * 3 - 1
        elif self.tails is None:
            return self.num_bins * 3 + 1
        else:
            raise ValueError

    def _elementwise(self, inputs, autoregressive_params, inverse=False):
        batch_size, features = inputs.shape[0], inputs.shape[1]

        transform_params = autoregressive_params.view(batch_size, features, self._output_dim_multiplier())

        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]

        if hasattr(self.autoregressive_net, "hidden_features"):
            unnormalized_widths /= np.sqrt(self.autoregressive_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.autoregressive_net.hidden_features)

        if self.tails is None:
            spline_fn = rational_quadratic_spline
            spline_kwargs = {}
        elif self.tails == "linear":
            spline_fn = unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}
        else:
            raise ValueError

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs,
        )

        return outputs, torchutils.sum_except_batch(logabsdet)

    def _elementwise_forward(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params)

    def _elementwise_inverse(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params, inverse=True)


class CompositeCDFTransform(CompositeTransform):
    def __init__(self, squashing_transform, cdf_transform):
        super().__init__(
            [
                squashing_transform,
                cdf_transform,
                InverseTransform(squashing_transform),
            ]
        )


class MaskedFixedNorm(Transform):
    def __init__(self, inputs, weights, mask, context_transform):
        super().__init__()
        self._dim = inputs.shape[1]
        self._scale = torch.ones(self._dim)
        self._shift = torch.zeros(self._dim)
        self._ct = context_transform  # TODO assert this is a lambda and equals mask given context
        if weights == None:
            n = inputs.shape[0]
            weights = torch.full([n], 1.0 / n)
        for d in range(self._dim):
            if mask[:, d].sum() == 0:
                self._shift[d] = 0
                self._scale[d] = 1
                continue
            self._shift[d] = (inputs[mask[:, d], d] @ weights[mask[:, d]]) / weights[
                mask[:, d]
            ].sum()  # torch.mean(inputs[:,d])
            try:
                cov = torch.Tensor(
                    np.cov(
                        inputs[mask[:, d], d].detach().numpy().flatten(),
                        aweights=weights[mask[:, d]].detach().numpy(),
                    )
                )  # workaround for torch 1.9.1
            except Exception:
                print(d)
                print(mask[:, d].sum())
                print(mask.shape)
                print(mask.sum())
                print(weights[mask[:, d]])
                sys.exit(0)
            self._scale[d] = cov**0.5
            self._scale[d] = max(self._scale[d], 1e-10)
            if torch.any(~torch.isfinite(self._scale)):
                print("inputs are singular, ", d)
                print(inputs[~torch.isfinite(inputs[mask[:, d], d])])
                print(self._scale)
                print(inputs)
                print(cov)
                print(weights)
                sys.exit(0)

    def inverse(self, inputs, context=None):
        if context is None:
            outputs = self._scale * inputs + self._shift
            return outputs, torch.log(torch.abs(self._scale * torch.ones_like(inputs))).sum(axis=-1)
        else:
            mask = self._ct(context)
            N = inputs.shape[0]
            outputs = inputs.clone()
            outputs[mask] = (torch.tile(self._scale, (N, 1)) * inputs + torch.tile(self._shift, (N, 1)))[mask]
            return outputs, (torch.log(torch.abs(torch.tile(self._scale, (N, 1)))) * mask.type(torch.int32)).sum(
                axis=-1
            )

    def forward(self, inputs, context=None):
        if context is None:
            outputs = 1.0 / self._scale * (inputs - self._shift)
            return outputs, -torch.log(torch.abs(self._scale * torch.ones_like(inputs))).sum(axis=-1)
        else:
            N = inputs.shape[0]
            mask = self._ct(context)
            outputs = inputs.clone()
            outputs[mask] = (torch.tile(self._scale ** (-1), (N, 1)) * (inputs - torch.tile(self._shift, (N, 1))))[mask]
            ld = (-torch.log(torch.abs(torch.tile(self._scale, (N, 1)))) * mask.type(torch.int32)).sum(axis=-1)
            return outputs, ld


class ConditionalMaskedTransform(Transform):
    def __init__(self, tf, context_transform):
        super().__init__()
        self._tf = tf  # todo assert transform
        self._ct = context_transform  # TODO assert this is a lambda and equals mask given context

    def forward(self, inputs, context=None):
        if context is None:
            return self._tf.forward(inputs)
        else:
            N = inputs.shape[0]
            mask = self._ct(context)
            outputs = inputs.clone()
            ld = torch.zeros_like(outputs)
            maskN = mask.sum()
            outputs_temp, ld_temp = self._tf.forward(inputs[mask].reshape((maskN, 1)))
            outputs[mask] = outputs_temp.reshape(maskN)
            ld[mask] = ld_temp.reshape(maskN)
            logabsdet = torchutils.sum_except_batch(ld, num_batch_dims=1)
            return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if context is None:
            return self._tf.inverse(inputs)
        else:
            N = inputs.shape[0]
            mask = self._ct(context)
            maskN = mask.sum()
            outputs = inputs.clone()
            ld = torch.zeros_like(outputs)
            outputs_temp, ld_temp = self._tf.inverse(inputs[mask].reshape((maskN, 1)))
            outputs[mask] = outputs_temp.reshape(maskN)
            ld[mask] = ld_temp.reshape(maskN)
            logabsdet = torchutils.sum_except_batch(ld, num_batch_dims=1)
            return outputs, logabsdet


class FixedLinear(Transform):
    def __init__(self, shift, scale):
        super().__init__()
        self._shift = shift
        self._scale = scale

    def inverse(self, inputs, context=None):
        outputs = 1.0 / self._scale * (inputs - self._shift)
        logabsdet = torchutils.sum_except_batch(
            -torch.log(torch.abs(self._scale * torch.ones_like(inputs))),
            num_batch_dims=1,
        )
        return outputs, logabsdet

    def forward(self, inputs, context=None):
        outputs = self._scale * inputs + self._shift
        ld_temp = torch.log(torch.abs(self._scale * torch.ones_like(inputs)))
        logabsdet = torchutils.sum_except_batch(
            torch.log(torch.abs(self._scale * torch.ones_like(inputs))),
            num_batch_dims=1,
        )
        return outputs, logabsdet
