"""Tests comparing mvgkde to jax.scipy."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays
from jax import Array
from jax.scipy.stats import gaussian_kde as jax_gaussian_kde

from mvgkde import gaussian_kde

dataset_strategy = arrays(
    dtype=np.float32,
    shape=st.tuples(
        st.integers(min_value=1, max_value=10),  # features
        st.integers(min_value=10, max_value=1000),  # samples
    ),
    elements=st.floats(-10, 10),
).filter(
    lambda x: np.all(np.std(x, axis=1) > 1e-1)  # avoid singular matrices
    and np.all(  # avoid only repeated values
        np.apply_along_axis(lambda col: len(np.unique(col)) > 5, axis=1, arr=x),
    ),
)

bw_method_strategy = st.one_of(
    st.just("scott"),  # Fixed value "scott"
    st.just("silverman"),  # Fixed value "silverman"
    st.just(None),  # Fixed value None
    st.floats(min_value=0.1, max_value=10),  # Scalar values
    # st.just( lambda dataset, weights: 1.0 ),  # noqa: ERA001
)


def generate_points(dataset: Array) -> Array:
    key = jr.fold_in(jr.key(0), hash(dataset.tobytes()) % (2**32))
    return jr.uniform(
        key,
        (dataset.shape[0], 100),
        dtype=dataset.dtype,
        minval=jnp.min(dataset, axis=1, keepdims=True),
        maxval=jnp.max(dataset, axis=1, keepdims=True),
    )


@settings(deadline=1_000)  # 1 s b/c of jit compilation
@given(dataset=dataset_strategy, bw=bw_method_strategy)
def test_evaluate(dataset, bw):
    kde = gaussian_kde(dataset, bw_method=bw)
    jax_kde = jax_gaussian_kde(dataset, bw_method=bw)

    points = generate_points(dataset)

    assert np.allclose(kde.evaluate(points), jax_kde(points), atol=1e-7)


@settings(deadline=1_000)  # 1 s b/c of jit compilation
@given(dataset=dataset_strategy, bw=bw_method_strategy)
def test_call(dataset, bw):
    kde = gaussian_kde(dataset, bw_method=bw)
    jax_kde = jax_gaussian_kde(dataset, bw_method=bw)

    points = generate_points(dataset)
    assert np.allclose(kde(points), jax_kde(points), atol=1e-7)


@settings(deadline=1_000)  # 1 s b/c of jit compilation
@given(dataset=dataset_strategy, bw=bw_method_strategy)
def test_pdf(dataset, bw):
    kde = gaussian_kde(dataset, bw_method=bw)
    jax_kde = jax_gaussian_kde(dataset, bw_method=bw)

    points = generate_points(dataset)
    assert np.allclose(kde.pdf(points), jax_kde(points), atol=1e-7)


@settings(deadline=1_000)  # 1 s b/c of jit compilation
@given(dataset=dataset_strategy, bw=bw_method_strategy)
def test_logpdf(dataset, bw):
    kde = gaussian_kde(dataset, bw_method=bw)
    jax_kde = jax_gaussian_kde(dataset, bw_method=bw)

    points = generate_points(dataset)
    assert np.allclose(kde.logpdf(points), jax_kde.logpdf(points), atol=1e-5)


@settings(deadline=1_000)  # 1 s b/c of jit compilation
@given(dataset=dataset_strategy, bw=bw_method_strategy)
def test_integrate_gaussian(dataset, bw):
    kde = gaussian_kde(dataset, bw_method=bw)
    jax_kde = jax_gaussian_kde(dataset, bw_method=bw)
    mean = jnp.zeros(dataset.shape[0])
    cov = jnp.eye(dataset.shape[0])

    result = kde.integrate_gaussian(mean, cov)
    jax_result = jax_kde.integrate_gaussian(mean, cov)

    assert np.allclose(result, jax_result, atol=1e-5)


@settings(deadline=1_000)  # 1 s b/c of jit compilation
@given(dataset=dataset_strategy, bw=bw_method_strategy)
def test_integrate_box_1d(dataset, bw):
    kde = gaussian_kde(dataset, bw_method=bw)
    jax_kde = jax_gaussian_kde(dataset, bw_method=bw)
    low, high = -5, 5

    if dataset.shape[0] != 1:
        with pytest.raises(ValueError, match="integrate_box_1d"):
            kde.integrate_box_1d(low, high)

        with pytest.raises(ValueError, match="integrate_box_1d"):
            jax_kde.integrate_box_1d(low, high)

        return

    result = kde.integrate_box_1d(low, high)
    jax_result = jax_kde.integrate_box_1d(low, high)
    assert np.allclose(result, jax_result, atol=1e-6)


@settings(deadline=1_000)  # 1 s b/c of jit compilation
@given(dataset=dataset_strategy, bw=bw_method_strategy)
def test_integrate_kde(dataset, bw):
    kde = gaussian_kde(dataset, bw_method=bw)
    jax_kde = jax_gaussian_kde(dataset, bw_method=bw)
    other_kde = gaussian_kde(dataset, bw_method=bw)
    jax_other_kde = jax_gaussian_kde(dataset, bw_method=bw)
    result = kde.integrate_kde(other_kde)
    jax_result = jax_kde.integrate_kde(jax_other_kde)
    assert np.allclose(result, jax_result, atol=1e-6)


@settings(deadline=1_000)  # 1 s b/c of jit compilation
@given(dataset=dataset_strategy, bw=bw_method_strategy)
def test_integrate_box(dataset, bw):
    kde = gaussian_kde(dataset, bw_method=bw)
    jax_kde = jax_gaussian_kde(dataset, bw_method=bw)

    low, high = -5, 5
    with pytest.raises(NotImplementedError):
        kde.integrate_box(low, high)

    with pytest.raises(NotImplementedError):
        jax_kde.integrate_box(low, high)


@settings(deadline=1_000)  # 1 s b/c of jit compilation
@given(dataset=dataset_strategy, bw=bw_method_strategy)
def test_resample(dataset, bw):
    kde = gaussian_kde(dataset, bw_method=bw)
    jax_kde = jax_gaussian_kde(dataset, bw_method=bw)

    key = jr.key(72)
    num_samples = (100,)

    samples = kde.resample(key, num_samples)
    jax_samples = jax_kde.resample(key, num_samples)

    assert samples.shape == jax_samples.shape
    assert np.allclose(samples, jax_samples, atol=1e-6)
