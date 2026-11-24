# Copyright (c) 2025 Nathaniel Starkman
# SPDX-License-Identifier: Apache-2.0

"""Test per-point kernels and the sample-point adaptive KDE."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.stats import multivariate_normal

from mvgkde import MultiVariateGaussianKDE


@pytest.fixture(autouse=True)
def x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)  # noqa: FBT003
    yield
    jax.config.update("jax_enable_x64", old)


def arc(key, n=200):
    """Sample a thin half-circle of radius 1 and width 0.02."""
    k1, k2 = jr.split(key)
    theta = jr.uniform(k1, (n,), minval=0, maxval=jnp.pi)
    r = 1 + 0.02 * jr.normal(k2, (n,))
    return jnp.stack([r * jnp.cos(theta), r * jnp.sin(theta)])


def on_arc(n=30):
    theta = jnp.linspace(0.1, 3.0, n)
    return jnp.stack([jnp.cos(theta), jnp.sin(theta)])


def test_equal_per_point_covariances_match_shared():
    data = arc(jr.key(0))
    n = data.shape[1]
    w = jr.uniform(jr.key(1), (n,), minval=0.5, maxval=1.5)
    cov = jnp.array([[0.3, 0.1], [0.1, 0.2]])
    shared = MultiVariateGaussianKDE.from_covariance(data, cov, weights=w)
    per_pt = MultiVariateGaussianKDE.from_covariance(
        data,
        jnp.broadcast_to(cov, (n, 2, 2)),
        weights=w,
    )
    pts = on_arc()
    mean, cov2 = jnp.array([0.1, 0.5]), jnp.array([[0.2, 0.05], [0.05, 0.1]])

    checks = [
        # batch_size 7 doesn't divide 30 points, so the remainder path runs too
        (shared.evaluate(pts), per_pt.evaluate(pts, batch_size=7)),
        (shared.logpdf(pts), per_pt.logpdf(pts)),
        (shared.resample(jr.key(2), (50,)), per_pt.resample(jr.key(2), (50,))),
        (shared.marginal(0).logpdf(pts[0]), per_pt.marginal(0).logpdf(pts[0])),
        (
            shared.marginal(jnp.array([1, 0])).evaluate(pts[::-1]),
            per_pt.marginal(jnp.array([1, 0])).evaluate(pts[::-1]),
        ),
        (shared.integrate_gaussian(mean, cov2), per_pt.integrate_gaussian(mean, cov2)),
        (
            shared.marginal(1).integrate_box_1d(0.2, 0.8),
            per_pt.marginal(1).integrate_box_1d(0.2, 0.8),
        ),
    ]
    for a, b in checks:
        np.testing.assert_allclose(b, a, rtol=1e-12, atol=1e-12)


def test_local_covariance_gradients_are_smooth():
    data = arc(jr.key(3))
    v = jr.normal(jr.key(4), data.shape)
    pts = on_arc()

    @jax.jit
    def f(t):
        kde = MultiVariateGaussianKDE.from_local_covariance(data + t * v)
        return jnp.sum(kde.logpdf(pts))

    # Along a random direction the gradient matches finite differences, at
    # several places. A kink anywhere within h of one of them would fail this.
    ts = jnp.linspace(-0.005, 0.005, 9)
    h = 1e-6
    g = jax.vmap(jax.grad(f))(ts)
    fd = (jax.vmap(f)(ts + h) - jax.vmap(f)(ts - h)) / (2 * h)
    assert jnp.all(jnp.isfinite(g))
    np.testing.assert_allclose(g, fd, rtol=1e-6)

    # A scan crossing many neighbour-order changes has no jumps. Hard top-k
    # neighbours give second differences above 0.1 on this scan.
    ts = jnp.linspace(-0.005, 0.005, 401)
    assert jnp.max(jnp.abs(jnp.diff(jax.vmap(f)(ts), 2))) < 1e-3


def test_zero_weight_points_change_no_kernel():
    data = arc(jr.key(5), n=100)
    w = jr.uniform(jr.key(6), (100,), minval=0.5, maxval=1.5)
    # 60 points on the arc would change k from round(sqrt(100)) to round(sqrt(160)).
    extra = 1.01 * on_arc(60)
    a = MultiVariateGaussianKDE.from_local_covariance(data, weights=w)
    b = MultiVariateGaussianKDE.from_local_covariance(
        jnp.concatenate([data, extra], axis=1),
        weights=jnp.concatenate([w, jnp.zeros(60)]),
    )
    np.testing.assert_allclose(b.covariance[:100], a.covariance, rtol=1e-12, atol=1e-15)
    assert jnp.all(jnp.isfinite(b.covariance))
    np.testing.assert_allclose(b.logpdf(on_arc()), a.logpdf(on_arc()), rtol=1e-12)


def test_local_kernels_follow_the_arc():
    data = arc(jr.key(7))
    kde = MultiVariateGaussianKDE.from_local_covariance(data)
    # Each kernel's long axis is along the tangent (-sin, cos) of the arc.
    _, vecs = jnp.linalg.eigh(kde.covariance)
    theta = jnp.arctan2(data[1], data[0])
    tangent = jnp.stack([-jnp.sin(theta), jnp.cos(theta)], axis=-1)
    alignment = jnp.abs(jnp.sum(vecs[..., :, -1] * tangent, axis=-1))
    assert jnp.median(alignment) > 0.99


def test_integrate_kde_rejects_per_point_kernels():
    kde = MultiVariateGaussianKDE.from_local_covariance(arc(jr.key(8)))
    with pytest.raises(NotImplementedError, match="per-point"):
        kde.integrate_kde(kde)


def test_gradient_memory_is_not_quadratic():
    def temp_bytes(n):
        data = arc(jr.key(9), n=n)

        def f(x):
            kde = MultiVariateGaussianKDE.from_local_covariance(x, batch_size=64)
            return jnp.sum(kde.logpdf(x, batch_size=64))

        compiled = jax.jit(jax.grad(f)).lower(data).compile()
        return compiled.memory_analysis().temp_size_in_bytes

    # N^2 memory would grow 4 times when N doubles.
    assert temp_bytes(2000) < 2.5 * temp_bytes(1000)


def test_float32_data_keeps_float32_kernels():
    data = arc(jr.key(10)).astype(jnp.float32)
    kde = MultiVariateGaussianKDE.from_local_covariance(data)
    assert kde.covariance.dtype == jnp.float32
    assert kde.logpdf(on_arc().astype(jnp.float32)).dtype == jnp.float32


def test_resample_uses_each_points_own_kernel():
    # A tight kernel at 0 and a wide one at 100: the samples near 0 must be tight,
    # also when the batch shape is square and its axes could be swapped.
    data = jnp.array([[0.0, 100.0], [0.0, 0.0]])
    cov = jnp.stack([1e-6 * jnp.eye(2), 100 * jnp.eye(2)])
    kde = MultiVariateGaussianKDE.from_covariance(data, cov)
    for shape in [(400,), (20, 20), (2, 10, 20)]:
        x = kde.resample(jr.key(11), shape)
        assert x.shape == (2, *shape)
        near = jnp.abs(x[0]) < 50
        assert jnp.std(jnp.where(near, x[0], 0)) < 1e-2


def test_local_covariance_second_derivatives():
    data = arc(jr.key(12), n=100)
    v = jr.normal(jr.key(13), data.shape)

    def f(t):
        return jnp.sum(
            MultiVariateGaussianKDE.from_local_covariance(data + t * v).covariance,
        )

    h = 1e-5
    fd = (jax.grad(f)(h) - jax.grad(f)(-h)) / (2 * h)
    np.testing.assert_allclose(jax.grad(jax.grad(f))(0.0), fd, rtol=1e-6)


def test_local_covariance_with_exact_duplicates():
    # A noisy ring with 17 copies of its centre, fewer than k = 18. The copies
    # move together, so they stay duplicates.
    k1, k2, k3 = jr.split(jr.key(14), 3)
    theta = jnp.linspace(0, 2 * jnp.pi, 300, endpoint=False)
    ring = jnp.stack([jnp.cos(theta), jnp.sin(theta)])
    ring = ring + 0.05 * jr.normal(k1, ring.shape)
    data = jnp.concatenate([ring, jnp.zeros((2, 17))], axis=1)
    v = jnp.concatenate(
        [jr.normal(k2, ring.shape), jnp.tile(jr.normal(k3, (2, 1)), (1, 17))],
        axis=1,
    )

    def f(t):
        kde = MultiVariateGaussianKDE.from_local_covariance(data + t * v)
        return jnp.sum(kde.covariance[-1])

    h = 1e-6
    np.testing.assert_allclose(jax.grad(f)(0.0), (f(h) - f(-h)) / (2 * h), rtol=1e-6)


def reference_kernels(data, w, k):
    """Compute the kernels of `from_local_covariance` with brentq and np.cov."""
    n, d = data.shape[1], data.shape[0]
    keep = w > 0
    metric = np.cov(data[:, keep], aweights=w[keep])
    y = np.linalg.solve(np.linalg.cholesky(metric), data).T
    out = []
    for i in range(n):
        others = keep & (np.arange(n) != i)
        d2 = np.sum((y - y[i]) ** 2, axis=-1)[others]
        log_d2 = np.log(d2[d2 > 0])

        def count(s, d2=d2):
            return np.sum(np.exp(-0.5 * d2 * np.exp(-s))) - k

        s = brentq(count, log_d2.min() - 10, log_d2.max() + 10, xtol=1e-14)
        g = w[others] * np.exp(-0.5 * d2 * np.exp(-s))
        out.append(k ** (-2 / (d + 4)) * np.cov(data[:, others], aweights=g))
    return np.array(out)


def test_local_covariance_matches_brute_force_reference():
    rng = np.random.default_rng(15)
    data = rng.normal(size=(2, 30)) * np.array([[3.0], [0.5]])
    data[1] += 0.3 * data[0] ** 2
    w = rng.uniform(0.2, 2.0, 30)
    w[[4, 17]] = 0.0
    kde = MultiVariateGaussianKDE.from_local_covariance(data, weights=w, k=4)
    ref = reference_kernels(data, w, 4)
    np.testing.assert_allclose(kde.covariance, ref, rtol=1e-10)

    # The density with these distinct kernels, summed directly.
    pts = rng.normal(size=(2, 7))
    direct = sum(
        w[j] / w.sum() * multivariate_normal(data[:, j], ref[j]).pdf(pts.T)
        for j in range(30)
    )
    np.testing.assert_allclose(kde.evaluate(pts, batch_size=3), direct, rtol=1e-10)
    np.testing.assert_allclose(kde.logpdf(pts), np.log(direct), rtol=1e-10)


def test_local_covariance_with_a_very_close_pair():
    # A pair 1e-145 apart, next to a far cluster, makes the bisection bracket
    # about 700 wide in log r^2.
    rng = np.random.default_rng(16)
    data = np.concatenate(
        [
            np.array([[0.0, 1e-145], [0.0, 0.0]]),
            rng.normal(size=(2, 1)),
            rng.normal(size=(2, 71)) + np.array([[210.0], [0.0]]),
        ],
        axis=1,
    )
    w = np.ones(data.shape[1])
    kde = MultiVariateGaussianKDE.from_local_covariance(data, k=2)
    np.testing.assert_allclose(
        kde.covariance,
        reference_kernels(data, w, 2),
        rtol=1e-10,
    )


def test_nan_kernel_of_a_zero_weight_point_keeps_the_density_finite():
    # The zero-weight copy of point 0 has 10 exact duplicates with weight, so
    # its own kernel is NaN, but it has no weight in the density.
    base = jr.normal(jr.key(17), (2, 100))
    data = jnp.concatenate([base, jnp.tile(base[:, :1], (1, 10))], axis=1)
    w = jnp.ones(110).at[-1].set(0.0)
    kde = MultiVariateGaussianKDE.from_local_covariance(data, weights=w)
    pts = on_arc(5)
    assert jnp.all(jnp.isfinite(kde.evaluate(pts)))
    np.testing.assert_allclose(jnp.log(kde.evaluate(pts)), kde.logpdf(pts), rtol=1e-12)


def test_weight_gradients_match_shared_path():
    # Includes a zero weight, where evaluate is still linear in the weight.
    data = jr.normal(jr.key(18), (2, 20))
    cov = jnp.array([[0.3, 0.1], [0.1, 0.2]])
    w = jnp.ones(20).at[3].set(0.0)
    pts = jr.normal(jr.key(19), (2, 3))

    def grads(c, method):
        def f(w):
            kde = MultiVariateGaussianKDE.from_covariance(data, c, weights=w)
            return jnp.sum(getattr(kde, method)(pts))

        return jax.grad(f)(w)

    per_pt = jnp.broadcast_to(cov, (20, 2, 2))
    g_shared, g_pp = grads(cov, "evaluate"), grads(per_pt, "evaluate")
    np.testing.assert_allclose(g_pp, g_shared, rtol=1e-12, atol=1e-15)
    pos = w > 0
    g_shared, g_pp = grads(cov, "logpdf"), grads(per_pt, "logpdf")
    np.testing.assert_allclose(g_pp[pos], g_shared[pos], rtol=1e-12, atol=1e-15)


def test_float32_kernels_far_from_the_origin():
    # Rounding must scale with the distances between points, not with |x|.
    data = (arc(jr.key(20)) + 1e4).astype(jnp.float32)
    k32 = MultiVariateGaussianKDE.from_local_covariance(data)
    k64 = MultiVariateGaussianKDE.from_local_covariance(data.astype(jnp.float64))
    np.testing.assert_allclose(k32.covariance, k64.covariance, rtol=1e-3, atol=1e-9)
