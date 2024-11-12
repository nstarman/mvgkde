"""Gaussian Kernel Density Estimator."""

# NOTE: This file modifies code from JAX, which is licensed under the Apache
# License 2.0. Copyright 2018 Google LLC.

__all__ = ["MultiVariateGaussianKDE", "gaussian_kde"]

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from typing import Annotated as Antd, Any, Literal, Never, TypeAlias, cast, final
from typing_extensions import Doc

import jax.numpy as jnp
import numpy as np
from jax import jit, lax, random, vmap
from jax._src.numpy.util import check_arraylike, promote_dtypes_inexact
from jax._src.prng import PRNGKeyArray
from jax.scipy import linalg, special
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float, Int

Scalar: TypeAlias = Float[Array, ""]
Neff: TypeAlias = Float[Array, ""] | float
Dataset: TypeAlias = Float[Array, "F N"] | Float[Array, "N"]
Weights: TypeAlias = Float[Array, "N"]
Cov: TypeAlias = Float[Array, "F F"]
Points: TypeAlias = Float[Array, "F"] | Float[Array, "F points"]
EvalOnPoint: TypeAlias = Float[Array, ""] | Float[Array, "point"]

BWMethodOpts = (
    Literal["scott", "silverman"] | Scalar | Callable[[Dataset, Weights], Scalar] | None
)


@final
@register_pytree_node_class
@dataclass(frozen=True, slots=True)
class MultiVariateGaussianKDE:
    """MultiVariate Gaussian Kernel Density Estimator.

    JAX re-implementation of `scipy.stats.gaussian_kde`.

    """

    dataset: Antd[
        Dataset,
        Doc(
            "Data for distribution estimation. If 1D, shape is (n_data,). "
            "If 2D, shape is (n_dimensions, n_data).",
        ),
    ]
    weights: Antd[Weights, Doc("Weights for data points. Same shape as dataset.")]
    neff: Antd[Neff, Doc("Effective number of data points.")]
    covariance: Antd[Cov, Doc("Covariance matrix.")]
    inv_cov: Antd[Cov, Doc("Inverse covariance matrix.")]

    def __post_init__(self) -> None:
        self._validate_dataset(self.dataset)
        self._validate_weights(self.weights, self.dataset.shape[1])

    # ====================================================================
    # Constructors

    @classmethod
    def from_covariance(
        cls,
        dataset: Dataset,
        /,
        covariance: Cov,
        *,
        weights: Float[Array, "N"] | None = None,
    ) -> "MultiVariateGaussianKDE":
        # Dataset
        dataset = jnp.atleast_2d(dataset)
        cls._validate_dataset(dataset)

        # Weights
        n = dataset.shape[1]
        if weights is not None:
            dataset, weights = promote_dtypes_inexact(dataset, weights)
            weights = jnp.atleast_1d(weights)
            weights /= jnp.sum(weights)
        else:
            (dataset,) = promote_dtypes_inexact(dataset)
            weights = jnp.full(n, 1.0 / n, dtype=dataset.dtype)

        return cls(
            dataset=dataset,
            weights=weights,
            neff=1 / jnp.sum(jnp.square(weights)),
            covariance=covariance,
            inv_cov=linalg.inv(covariance),
        )

    @classmethod
    def from_bandwidth(
        cls,
        dataset: Dataset,
        /,
        *,
        bw: BWMethodOpts = None,
        weights: Float[Array, "N"] | None = None,
    ) -> "MultiVariateGaussianKDE":
        # Dataset
        dataset = jnp.atleast_2d(dataset)
        cls._validate_dataset(dataset)

        # Weights
        n = dataset.shape[1]
        if weights is not None:
            dataset, weights = promote_dtypes_inexact(dataset, weights)
            weights = jnp.atleast_1d(weights)
            weights /= jnp.sum(weights)
        else:
            (dataset,) = promote_dtypes_inexact(dataset)
            weights = jnp.full(n, 1.0 / n, dtype=dataset.dtype)
        cls._validate_weights(weights, n)

        # Neff
        neff = 1 / jnp.sum(jnp.square(weights))

        # Covariance
        cov, inv_cov = cls._parse_data_covariance(dataset, weights, neff, bw)

        return cls(
            dataset=dataset,
            weights=weights,
            neff=neff,
            covariance=cov,
            inv_cov=inv_cov,
        )

    # ====================================================================
    # Properties

    @property
    def d(self) -> int:
        """Number of dimensions (features) in the dataset."""
        return cast(int, self.dataset.shape[0])

    @property
    def n(self) -> int:
        """Number of data points in the dataset."""
        return cast(int, self.dataset.shape[1])

    # ====================================================================
    # Evaluation

    def evaluate(
        self,
        points: Points,
    ) -> Antd[EvalOnPoint, Doc("The values at each point.")]:
        """Evaluate the estimated pdf on a set of points."""
        check_arraylike("evaluate", points)
        points = self._reshape_points(points)
        result = _gaussian_kernel_eval(
            self.dataset.T,
            self.weights[:, None],
            points.T,
            self.inv_cov,
            in_log=False,
        )
        return result[:, 0]

    __call__ = evaluate
    pdf = evaluate

    def logpdf(self, x: Points) -> EvalOnPoint:
        """Log probability density function."""
        check_arraylike("logpdf", x)
        x = self._reshape_points(x)
        result = _gaussian_kernel_eval(
            self.dataset.T,
            self.weights[:, None],
            x.T,
            self.inv_cov,
            in_log=True,
        )
        return result[:, 0]

    def resample(
        self,
        key: Antd[PRNGKeyArray, Doc("PRNG key used as the random key.")],
        /,
        shape: Antd[
            tuple[int, ...],
            Doc(
                "A tuple of nonnegative integers specifying the result batch shape;"
                "that is, the prefix of the result shape excluding the last axis.",
            ),
        ] = (),
    ) -> Antd[Float[Array, "F {shape}"], Doc("The resampled dataset.")]:
        """Randomly sample a dataset from the estimated pdf.

        Note
        ----
        This method has a different signature than the `scipy.stats.gaussian_kde`.

        """
        ind_key, eps_key = random.split(key)
        ind = random.choice(ind_key, self.n, shape=shape, p=self.weights)
        eps = random.multivariate_normal(
            eps_key,
            jnp.zeros(self.d, self.covariance.dtype),
            self.covariance,
            shape=shape,
            dtype=self.dataset.dtype,
        ).T
        return self.dataset[:, ind] + eps

    def marginal(
        self,
        dimensions: Antd[
            int | Int[Array, "subF"],
            Doc(
                "The dimensions of the multivariate distribution corresponding "
                "with the marginal variables, that is, the indices of the dimensions "
                "that are being retained. The other dimensions are marginalized out.",
            ),
        ],
    ) -> Antd["MultiVariateGaussianKDE", Doc("The marginal distribution.")]:
        """Return a marginal distribution over a subset of the dimensions.

        Note
        ----
        This implementation differs from the `scipy.stats.gaussian_kde`, which
        will recompute the covariance matrix for the marginal distribution. This
        implementation retains the original covariance matrix, slicing it to
        match the dataset's features.

        """
        dims = jnp.atleast_1d(dimensions)

        if not jnp.issubdtype(dims.dtype, jnp.integer):
            msg = "Elements of `dimensions` must be integers."
            raise ValueError(msg)

        original_dims = dims.copy()

        neg_idx = dims < 0
        dims = dims.at[neg_idx].set(self.d + dims[neg_idx])

        if len(jnp.unique(dims)) != len(dims):
            msg = "All elements of `dimensions` must be unique."
            raise ValueError(msg)

        if jnp.any(i_invalid := (dims < 0) | (dims >= self.d)):
            msg = (
                f"Dimensions {original_dims[i_invalid]} are invalid "
                f"for a distribution in {self.d} dimensions."
            )
            raise ValueError(msg)

        new_cov = self.covariance[dims][:, dims]
        return replace(  # (weights, neff are unchanged)
            self,
            dataset=self.dataset[dims],
            covariance=new_cov,
            inv_cov=linalg.inv(new_cov),
        )

    # ====================================================================
    # Integration

    def integrate_gaussian(
        self,
        mean: Antd[Float[Array, "F"], Doc("The mean of the Gaussian.")],
        cov: Antd[Float[Array, "F F"], Doc("The covariance of the Gaussian.")],
    ) -> Antd[Scalar, Doc("The value of the integral.")]:
        """Integrate the distribution weighted by a multivariate Gaussian."""
        mean = jnp.atleast_1d(jnp.squeeze(mean))

        cov = jnp.atleast_2d(cov)

        if mean.shape != (self.d,):
            msg = f"mean does not have dimension {self.d}"
            raise ValueError(msg)
        if cov.shape != (self.d, self.d):
            msg = f"covariance does not have dimension {self.d}"
            raise ValueError(msg)

        chol = linalg.cho_factor(self.covariance + cov)
        norm = jnp.sqrt(2 * np.pi) ** self.d * jnp.prod(jnp.diag(chol[0]))
        norm = 1.0 / norm
        return _gaussian_kernel_convolve(chol, norm, self.dataset, self.weights, mean)

    def integrate_box_1d(
        self,
        low: Antd[Scalar, Doc("lower bound of integration.")],
        high: Antd[Scalar, Doc("upper bound of integration.")],
    ) -> Antd[Scalar, Doc("The value of the integral.")]:
        """Compute the integral of a 1D pdf between two bounds.."""
        if self.d != 1:
            msg = "integrate_box_1d() only handles 1D pdfs"
            raise ValueError(msg)
        if jnp.ndim(low) != 0 or jnp.ndim(high) != 0:
            msg = "the limits of integration in integrate_box_1d must be scalars"
            raise ValueError(msg)
        sigma = jnp.squeeze(jnp.sqrt(self.covariance))
        low = jnp.squeeze((low - self.dataset) / sigma)
        high = jnp.squeeze((high - self.dataset) / sigma)
        return jnp.sum(self.weights * (special.ndtr(high) - special.ndtr(low)))

    def integrate_box(
        self,
        low_bounds: Antd[Any, Doc("lower bounds of integration.")],
        high_bounds: Antd[Any, Doc("upper bounds of integration.")],
        maxpts: Antd[Any, Doc("Maximum number of points for integration.")] = None,
    ) -> Never:
        """This method is not implemented in the JAX interface."""  # noqa: D401, D404
        del low_bounds, high_bounds, maxpts
        msg = "only 1D box integrations are supported; use `integrate_box_1d`"
        raise NotImplementedError(msg)

    def integrate_kde(
        self,
        other: Antd["MultiVariateGaussianKDE", Doc("The other KDE.")],
    ) -> Antd[Scalar, Doc("The value of the integral.")]:
        """Integrate the product of two Gaussian KDE distributions."""
        if other.d != self.d:
            msg = "KDEs are not the same dimensionality"
            raise ValueError(msg)

        chol = linalg.cho_factor(self.covariance + other.covariance)
        norm = jnp.sqrt(2 * np.pi) ** self.d * jnp.prod(jnp.diag(chol[0]))
        norm = 1.0 / norm

        sm, lg = (self, other) if self.n < other.n else (other, self)
        result = vmap(
            partial(_gaussian_kernel_convolve, chol, norm, lg.dataset, lg.weights),
            in_axes=1,
        )(sm.dataset)
        return jnp.sum(result * sm.weights)

    # ====================================================================

    def set_bandwidth(self, bw_method: Any = None) -> Never:
        """This method is not implemented in the JAX interface."""  # noqa: D401, D404
        del bw_method
        msg = "dynamically changing the bandwidth method is not supported"
        raise NotImplementedError(msg)

    # ====================================================================
    # Parsing and Validation

    @staticmethod
    def _validate_dataset(dataset: Dataset) -> None:
        if jnp.issubdtype(lax.dtype(dataset), jnp.complexfloating):
            msg = "gaussian_kde does not support complex data"
            raise NotImplementedError(msg)

        if not dataset.size > 1:
            msg = "`dataset` input should have multiple elements."
            raise ValueError(msg)

    @staticmethod
    def _validate_weights(weights: Weights, n: int) -> None:
        if weights.ndim != 1:
            msg = "`weights` input should be one-dimensional."
            raise ValueError(msg)

        if len(weights) != n:
            msg = "`weights` input should be of length n"
            raise ValueError(msg)

    @staticmethod
    def _parse_data_covariance(
        dataset: Dataset,
        weights: Weights,
        neff: Neff,
        bw: BWMethodOpts,
    ) -> tuple[Cov, Cov]:
        d = dataset.shape[0]

        # Bandwidth
        #   string options: scott / None, silverman
        if bw == "scott" or bw is None:
            factor = scotts_factor(neff, d)
        elif bw == "silverman":
            factor = silverman_factor(neff, d)
        elif isinstance(bw, str):
            msg = "`bw` str options are {'scott', 'silverman'}."
            raise ValueError(msg)

        #   Callable
        elif callable(bw):
            factor = bw(dataset, weights)

        #   Array(-like) Options
        elif jnp.isscalar(bw):
            factor = bw
        elif hasattr(bw, "shape"):
            shape = bw.shape
            if shape == (d,):  # TODO: should this scale the off-diagonal?
                factor = jnp.eye(d) * bw
            elif shape == (d, d):
                factor = bw
            else:
                msg = f"`bw` array shape should be ({d},) or ({d}, {d})."
                raise ValueError(msg)

        else:
            msg = "`bw` should be 'scott', 'silverman', a scalar, or a callable."
            raise ValueError(msg)

        # Covariance
        data_covariance = jnp.cov(dataset, rowvar=True, bias=False, aweights=weights)
        covariance = jnp.atleast_2d(data_covariance) * factor**2
        inv_cov = linalg.inv(covariance)

        return covariance, inv_cov

    # ====================================================================
    # Helper methods

    def _reshape_points(self, points: Points) -> Float[Array, "F points"]:
        if jnp.issubdtype(lax.dtype(points), jnp.complexfloating):
            msg = "gaussian_kde does not support complex coordinates"
            raise NotImplementedError(msg)

        points = jnp.atleast_2d(points)
        d, m = points.shape
        if d != self.d:
            if d == 1 and m == self.d:
                points = jnp.reshape(points, (self.d, 1))
            else:
                msg = f"points have dimension {d}, dataset has dimension {self.d}"
                raise ValueError(msg)

        return points

    # ====================================================================
    # JAX PyTree methods

    def tree_flatten(self) -> tuple[tuple[Dataset, Weights, Neff, Cov, Cov], None]:
        """Flatten a Gaussian KDE into a Pytree representation."""
        return (
            (self.dataset, self.weights, self.neff, self.covariance, self.inv_cov),
            None,
        )

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: Antd[Any, Doc("Unused auxiliary data.")],
        children: tuple[Dataset, Weights, Neff, Cov, Cov],
    ) -> "MultiVariateGaussianKDE":
        """Reconstruct a Gaussian KDE from its Pytree representation."""
        del aux_data
        kde = cls(
            dataset=children[0],
            weights=children[1],
            neff=children[2],
            covariance=children[3],
            inv_cov=children[4],
        )
        return kde  # noqa: RET504


#######################################################################
# Scipy API


def gaussian_kde(
    dataset: Dataset,
    bw_method: BWMethodOpts = None,
    weights: Weights | None = None,
) -> MultiVariateGaussianKDE:
    """Gaussian Kernel Density Estimator.

    JAX implementation of `scipy.stats.gaussian_kde`.

    :param dataset: Data from which to estimate the distribution. If 1D, shape
        is (n_data,). If 2D, shape is (n_dimensions, n_data).
    :param bw_method: Either "scott", "silverman", a scalar value, or a callable
        function which takes ``self`` as a parameter.
    :param weights: Weights of the same shape as the dataset.

    :return: A Gaussian KDE object.

    """
    return MultiVariateGaussianKDE.from_bandwidth(
        dataset,
        bw=bw_method,
        weights=weights,
    )


# ====================================================================
# Gaussian Kernel Utils


def _gaussian_kernel_convolve(
    chol: Array,
    norm: Array,
    target: Float[Array, "F N"],
    weights: Weights,
    mean: Float[Array, "F"],
) -> Float[Array, "N"]:
    diff = target - mean[:, None]
    alpha = linalg.cho_solve(chol, diff)
    arg = 0.5 * jnp.sum(diff * alpha, axis=0)
    return norm * jnp.sum(jnp.exp(-arg) * weights)


@partial(jit, static_argnames=("in_log",))
def _gaussian_kernel_eval(
    points: Float[Array, "data F"],
    values: Float[Array, "N 1"],
    xi: Float[Array, "data F"],
    invcov: Cov,
    *,
    in_log: bool,
) -> Float[Array, "data"]:
    points, values, xi, invcov = promote_dtypes_inexact(points, values, xi, invcov)
    d = points.shape[1]

    if xi.shape[1] != d:
        msg = "points and xi must have same trailing dim"
        raise ValueError(msg)
    if invcov.shape != (d, d):
        msg = "precision matrix must match data dims"
        raise ValueError(msg)

    whitening = linalg.cholesky(invcov, lower=True)
    points = jnp.dot(points, whitening)
    xi = jnp.dot(xi, whitening)
    log_norm = jnp.sum(jnp.log(jnp.diag(whitening))) - 0.5 * d * jnp.log(2 * np.pi)

    def kernel(x_test: Array, x_train: Array, y_train: Array) -> Array:
        arg = log_norm - 0.5 * jnp.sum(jnp.square(x_train - x_test))
        if in_log:
            return jnp.log(y_train) + arg
        return y_train * jnp.exp(arg)

    reduce = special.logsumexp if in_log else jnp.sum
    reduced_kernel = lambda x: reduce(  # noqa: E731
        vmap(kernel, in_axes=(None, 0, 0))(x, points, values),
        axis=0,
    )
    mapped_kernel = vmap(reduced_kernel)

    return mapped_kernel(xi)


# ====================================================================


def scotts_factor(
    neff: Antd[Neff, Doc("Effective number of data points")],
    features: Antd[int, Doc("Dimension of the datasets")],
    /,
) -> Scalar:
    """Scott's factor for Gaussian KDE."""
    return jnp.power(neff, -1.0 / (features + 4))


def silverman_factor(
    neff: Antd[Neff, Doc("Effective number of data points")],
    features: Antd[int, Doc("Dimension of the datasets")],
    /,
) -> Scalar:
    """Silverman's factor for Gaussian KDE."""
    return jnp.power(neff * (features + 2) / 4.0, -1.0 / (features + 4))
