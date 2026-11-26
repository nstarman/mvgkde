"""Gaussian Kernel Density Estimator."""

# NOTE: This file modifies code from JAX, which is licensed under the Apache
# License 2.0. Copyright 2018 Google LLC.

__all__ = ["MultiVariateGaussianKDE", "gaussian_kde"]

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Literal, Never, TypeAlias, cast, final

import jax
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
Cov: TypeAlias = Float[Array, "F F"] | Float[Array, "N F F"]
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

    Attributes
    ----------
    dataset
        Data for distribution estimation. If 1D, shape is (n_data,). If 2D,
        shape is (n_dimensions, n_data).
    weights
        Weights for data points. Same shape as dataset.
    neff
        Effective number of data points.
    covariance
        Kernel covariance matrix. Shape (F, F) for one kernel shared by every
        data point, or (N, F, F) for one kernel per data point.
    inv_cov
        Inverse covariance matrix, same shape.

    """

    dataset: Dataset
    weights: Weights
    neff: Neff
    covariance: Cov
    inv_cov: Cov

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

    @classmethod
    def from_local_covariance(
        cls,
        dataset: Dataset,
        /,
        *,
        k: int | None = None,
        weights: Float[Array, "N"] | None = None,
        batch_size: int = 128,
    ) -> "MultiVariateGaussianKDE":
        """Build a sample-point adaptive KDE, with one kernel per data point.

        Each point's kernel is ``b**2`` times the weighted covariance of its
        neighbours, with ``b`` Scott's factor for ``k`` points. Neighbour ``j``
        of point ``i`` has weight ``w_j exp(-d_ij**2 / (2 r_i**2))``, and
        ``r_i`` is set so that ``exp(-d_ij**2 / (2 r_i**2))`` summed over the
        other points with weight is ``k``. This ``r_i`` is a smooth version of
        the distance to the ``k``-th nearest neighbour, so the density and its
        gradients have no jumps where the neighbour order changes. Distances
        are Mahalanobis distances in the dataset's covariance, so the result
        does not depend on the units of each dimension.

        Points with zero weight do not change any kernel, nor the default
        ``k = round(sqrt(n))``, where ``n`` counts points with positive weight.
        Each kernel needs about ``F + 1`` effective neighbours to be full rank,
        so ``n`` should be at least ``F + 2`` and ``k`` not much below ``F``.
        ``k`` must be less than ``n - 1``, so the default needs ``n >= 4``, and
        no point may have ``k`` exact duplicates. Otherwise the kernels are NaN.
        Rounded or binned data can easily have that many duplicates.

        The weights scale each neighbour's contribution, but ``r_i`` and the
        default ``k`` only count which points have positive weight. So the
        kernels are smooth in positive weights, and jump when a weight moves
        between zero and a positive value.

        ``batch_size`` points are processed at a time, so memory scales as
        ``batch_size * N * F``, also under ``jax.grad``.

        Examples
        --------
        >>> import jax.random as jr
        >>> from mvgkde import MultiVariateGaussianKDE
        >>> dataset = jr.normal(jr.key(0), (2, 200))
        >>> kde = MultiVariateGaussianKDE.from_local_covariance(dataset)
        >>> kde.covariance.shape
        (200, 2, 2)

        """
        kde = cls.from_bandwidth(dataset, weights=weights)
        # The metric's scale doesn't matter. Shifting to a data point first keeps
        # float32 rounding independent of how far the data are from the origin.
        offsets = kde.dataset - kde.dataset[:, :1]
        metric = jnp.atleast_2d(jnp.cov(offsets, aweights=kde.weights))
        cov = _local_covariance(
            kde.dataset,
            kde.weights,
            metric,
            k=k,
            batch_size=batch_size,
        )
        return replace(kde, covariance=cov, inv_cov=linalg.inv(cov))

    # ====================================================================
    # Properties

    @property
    def d(self) -> int:
        """Number of dimensions (features) in the dataset."""
        return cast("int", self.dataset.shape[0])

    @property
    def n(self) -> int:
        """Number of data points in the dataset."""
        return cast("int", self.dataset.shape[1])

    # ====================================================================
    # Evaluation

    def evaluate(
        self,
        points: Points,
        *,
        batch_size: int = 128,
    ) -> EvalOnPoint:
        """Evaluate the estimated pdf on a set of points.

        Parameters
        ----------
        points
            The points to evaluate at.
        batch_size
            Points evaluated at once with per-point kernels. A Python int; mark
            it static if you jit this method.

        Returns
        -------
        EvalOnPoint
            The values at each point.

        """
        check_arraylike("evaluate", points)
        points = self._reshape_points(points)
        if self.covariance.ndim == 3:
            return _per_point_kernel_eval(
                self.dataset.T,
                self.weights,
                points.T,
                self.inv_cov,
                in_log=False,
                batch_size=batch_size,
            )
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

    def logpdf(
        self,
        x: Points,
        *,
        batch_size: int = 128,
    ) -> EvalOnPoint:
        """Log probability density function.

        Parameters
        ----------
        x
            The points to evaluate at.
        batch_size
            Points evaluated at once with per-point kernels. A Python int; mark
            it static if you jit this method.

        """
        check_arraylike("logpdf", x)
        x = self._reshape_points(x)
        if self.covariance.ndim == 3:
            return _per_point_kernel_eval(
                self.dataset.T,
                self.weights,
                x.T,
                self.inv_cov,
                in_log=True,
                batch_size=batch_size,
            )
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
        key: PRNGKeyArray,
        /,
        shape: tuple[int, ...] = (),
    ) -> Float[Array, "F {shape}"]:
        """Randomly sample a dataset from the estimated pdf.

        Parameters
        ----------
        key
            PRNG key used as the random key.
        shape
            A tuple of nonnegative integers specifying the result batch shape;
            the result has shape (F, *shape).

        Returns
        -------
        Array
            The resampled dataset.

        Note
        ----
        This method has a different signature than the `scipy.stats.gaussian_kde`.

        """
        ind_key, eps_key = random.split(key)
        ind = random.choice(ind_key, self.n, shape=shape, p=self.weights)
        cov = self.covariance[ind] if self.covariance.ndim == 3 else self.covariance
        eps = random.multivariate_normal(
            eps_key,
            jnp.zeros(self.d, self.covariance.dtype),
            cov,
            shape=shape,
            dtype=self.dataset.dtype,
        )
        return self.dataset[:, ind] + jnp.moveaxis(eps, -1, 0)

    def marginal(
        self,
        dimensions: int | Int[Array, "subF"],
    ) -> "MultiVariateGaussianKDE":
        """Return a marginal distribution over a subset of the dimensions.

        Parameters
        ----------
        dimensions
            The dimensions of the multivariate distribution corresponding with
            the marginal variables, that is, the indices of the dimensions that
            are being retained. The other dimensions are marginalized out.

        Returns
        -------
        MultiVariateGaussianKDE
            The marginal distribution.

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

        new_cov = self.covariance[..., dims[:, None], dims]
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
        mean: Float[Array, "F"],
        cov: Float[Array, "F F"],
    ) -> Scalar:
        """Integrate the distribution weighted by a multivariate Gaussian.

        Parameters
        ----------
        mean
            The mean of the Gaussian.
        cov
            The covariance of the Gaussian.

        Returns
        -------
        Scalar
            The value of the integral.

        """
        mean = jnp.atleast_1d(jnp.squeeze(mean))

        cov = jnp.atleast_2d(cov)

        if mean.shape != (self.d,):
            msg = f"mean does not have dimension {self.d}"
            raise ValueError(msg)
        if cov.shape != (self.d, self.d):
            msg = f"covariance does not have dimension {self.d}"
            raise ValueError(msg)

        if self.covariance.ndim == 3:  # sum_j w_j N(mean; x_j, C_j + cov)
            return _per_point_kernel_eval(
                self.dataset.T,
                self.weights,
                mean[None],
                linalg.inv(self.covariance + cov),
                in_log=False,
                batch_size=1,
            )[0]

        chol = linalg.cho_factor(self.covariance + cov)
        norm = jnp.sqrt(2 * np.pi) ** self.d * jnp.prod(jnp.diag(chol[0]))
        norm = 1.0 / norm
        return _gaussian_kernel_convolve(chol, norm, self.dataset, self.weights, mean)

    def integrate_box_1d(
        self,
        low: Scalar,
        high: Scalar,
    ) -> Scalar:
        """Compute the integral of a 1D pdf between two bounds.

        Parameters
        ----------
        low
            Lower bound of integration.
        high
            Upper bound of integration.

        Returns
        -------
        Scalar
            The value of the integral.

        """
        if self.d != 1:
            msg = "integrate_box_1d() only handles 1D pdfs"
            raise ValueError(msg)
        if jnp.ndim(low) != 0 or jnp.ndim(high) != 0:
            msg = "the limits of integration in integrate_box_1d must be scalars"
            raise ValueError(msg)
        sigma = jnp.squeeze(jnp.sqrt(self.covariance))
        low = jnp.squeeze((low - self.dataset) / sigma)
        high = jnp.squeeze((high - self.dataset) / sigma)
        mass = special.ndtr(high) - special.ndtr(low)
        # As in `evaluate`: a zero-weight point's kernel may be NaN.
        mass = jnp.where(jnp.isfinite(mass) | (self.weights > 0), mass, 0)
        return jnp.sum(self.weights * mass)

    def integrate_box(
        self,
        low_bounds: Any,
        high_bounds: Any,
        maxpts: Any = None,
    ) -> Never:
        """This method is not implemented in the JAX interface.

        Parameters
        ----------
        low_bounds
            Lower bounds of integration.
        high_bounds
            Upper bounds of integration.
        maxpts
            Maximum number of points for integration.

        """  # noqa: D401, D404
        del low_bounds, high_bounds, maxpts
        msg = "only 1D box integrations are supported; use `integrate_box_1d`"
        raise NotImplementedError(msg)

    def integrate_kde(
        self,
        other: "MultiVariateGaussianKDE",
    ) -> Scalar:
        """Integrate the product of two Gaussian KDE distributions.

        Parameters
        ----------
        other
            The other KDE.

        Returns
        -------
        Scalar
            The value of the integral.

        """
        if other.d != self.d:
            msg = "KDEs are not the same dimensionality"
            raise ValueError(msg)
        if self.covariance.ndim == 3 or other.covariance.ndim == 3:
            msg = "integrate_kde does not support per-point kernels"
            raise NotImplementedError(msg)

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
        aux_data: Any,
        children: tuple[Dataset, Weights, Neff, Cov, Cov],
    ) -> "MultiVariateGaussianKDE":
        """Reconstruct a Gaussian KDE from its Pytree representation.

        Parameters
        ----------
        aux_data
            Unused auxiliary data.
        children
            The flattened fields, as `tree_flatten` returns them.

        """
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


@partial(jit, static_argnames=("in_log", "batch_size"))
def _per_point_kernel_eval(
    points: Float[Array, "N F"],
    values: Float[Array, "N"],
    xi: Float[Array, "M F"],
    invcov: Float[Array, "N F F"],
    *,
    in_log: bool,
    batch_size: int,
) -> Float[Array, "M"]:
    """Evaluate like `_gaussian_kernel_eval`, with one precision per point."""
    points, values, xi, invcov = promote_dtypes_inexact(points, values, xi, invcov)
    d = points.shape[1]

    whitening = jnp.linalg.cholesky(invcov)  # (N, F, F)
    log_norm = jnp.sum(
        jnp.log(jnp.diagonal(whitening, axis1=-2, axis2=-1)),
        axis=-1,
    ) - 0.5 * d * jnp.log(2 * np.pi)

    def one(x: Float[Array, "F"]) -> Scalar:
        z = jnp.einsum("nf,nfg->ng", points - x, whitening)
        arg = log_norm - 0.5 * jnp.sum(jnp.square(z), axis=-1)
        # A zero-weight point's kernel may be NaN; drop only those terms, so the
        # gradient with respect to a zero weight is kept.
        arg = jnp.where(jnp.isfinite(arg) | (values > 0), arg, -jnp.inf)
        if in_log:  # as `_gaussian_kernel_eval`
            return special.logsumexp(jnp.log(values) + arg)
        return jnp.sum(values * jnp.exp(arg))

    # checkpoint, so reverse mode keeps batch_size * N * F values, not M * N * F
    return lax.map(jax.checkpoint(one), xi, batch_size=batch_size)


@partial(jit, static_argnames=("batch_size",))
def _local_covariance(
    dataset: Float[Array, "F N"],
    weights: Weights,
    metric: Float[Array, "F F"],
    *,
    k: int | Int[Array, ""] | None,
    batch_size: int,
) -> Float[Array, "N F F"]:
    """Compute per-point kernel covariances, see `from_local_covariance`."""
    d, n = dataset.shape
    keep = weights > 0
    # k is a float of the data's dtype, so it doesn't promote float32 data.
    k = jnp.round(jnp.sqrt(jnp.sum(keep, dtype=dataset.dtype))) if k is None else k
    k = jnp.asarray(k, dtype=dataset.dtype)

    # Distances are measured after whitening by ``metric``. Everything below is
    # in offsets from point i, so float32 rounding scales with the distances
    # between points, not with how far they are from the origin.
    chol = linalg.cholesky(metric, lower=True)
    x = dataset.T

    def one(args: tuple[Float[Array, "F"], Int[Array, ""]]) -> Float[Array, "F F"]:
        xi, i = args
        others = keep & (jnp.arange(n) != i)
        dx = x - xi
        y = linalg.solve_triangular(chol, dx.T, lower=True)
        d2 = jnp.sum(jnp.square(y), axis=0)

        # log r^2 solves count(log r^2) = k: the Gaussian factors of the other
        # points with weight sum to k. This is a smooth version of the distance
        # to the k-th nearest neighbour, which has kinks where the order changes.
        def count(s: Scalar, d2: Array) -> tuple[Scalar, Scalar]:
            g = jnp.where(others, jnp.exp(-0.5 * d2 * jnp.exp(-s)), 0)
            return jnp.sum(g) - k, jnp.sum(g * 0.5 * d2 * jnp.exp(-s))

        def newton(s: Scalar, d2: Array) -> Scalar:
            c, dc = count(s, d2)
            return s - c / dc

        # Bisect without gradients. At the ends of the bracket every other point
        # that isn't an exact duplicate is as far as the farthest, or as near as
        # the nearest, and the count is still at least, or at most, k.
        d2_ = lax.stop_gradient(d2)

        def bisect(b: tuple[Scalar, Scalar]) -> tuple[Scalar, Scalar]:
            mid = (b[0] + b[1]) / 2
            low = count(mid, d2_)[0] < 0
            return jnp.where(low, mid, b[0]), jnp.where(low, b[1], mid)

        pos = others & (d2_ > 0)
        log_d2 = jnp.log(jnp.where(pos, d2_, 1))
        # Exact duplicates add 1 to the count at every r.
        m = jnp.sum(others, dtype=dataset.dtype)
        n0 = jnp.sum(others & (d2_ == 0), dtype=dataset.dtype)
        shift = jnp.log(2 * jnp.log((m - n0) / (k - n0)))
        # Bisect until the bracket is 1e-3 wide; a very close pair can make it
        # hundreds wide. A NaN bracket (outside the documented limits) stops.
        lo, hi = lax.while_loop(
            lambda b: b[1] - b[0] > 1e-3,
            bisect,
            (
                jnp.min(jnp.where(pos, log_d2, jnp.inf)) - shift,
                jnp.max(jnp.where(pos, log_d2, -jnp.inf)) - shift,
            ),
        )
        # Polish with Newton. Only the last two steps see d2. One step gives the
        # exact first derivative of the root, two give exact second derivatives.
        s = newton((lo + hi) / 2, d2_)
        s = newton(newton(s, d2), d2)

        w = jnp.where(others, weights * jnp.exp(-0.5 * d2 * jnp.exp(-s)), 0)
        v1, v2 = jnp.sum(w), jnp.sum(jnp.square(w))
        dx = dx - w @ dx / v1
        return jnp.einsum("n,ni,nj->ij", w, dx, dx) / (v1 - v2 / v1)  # as jnp.cov

    # checkpoint, so reverse mode keeps batch_size * N * F values, not N * N * F
    cov = lax.map(jax.checkpoint(one), (x, jnp.arange(n)), batch_size=batch_size)
    return scotts_factor(k, d) ** 2 * cov


# ====================================================================


def scotts_factor(neff: Neff, features: int, /) -> Scalar:
    """Scott's factor for Gaussian KDE.

    Parameters
    ----------
    neff
        Effective number of data points.
    features
        Dimension of the datasets.

    """
    return jnp.power(neff, -1.0 / (features + 4))


def silverman_factor(neff: Neff, features: int, /) -> Scalar:
    """Silverman's factor for Gaussian KDE.

    Parameters
    ----------
    neff
        Effective number of data points.
    features
        Dimension of the datasets.

    """
    return jnp.power(neff * (features + 2) / 4.0, -1.0 / (features + 4))
