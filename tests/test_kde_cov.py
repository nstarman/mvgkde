"""Test the per-dimension covariance functionality."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from mvgkde import MultiVariateGaussianKDE

key = jr.key(42)


def generate_dataset(key, cov_matrix, num_samples=1_000):
    mean = jnp.zeros(cov_matrix.shape[0])
    return jr.multivariate_normal(key, mean, cov_matrix, (num_samples,)).T


@pytest.fixture
def kde_fixture():
    # Define the covariance matrix and generate the dataset
    cov_matrix = jnp.array([[1, 0], [0, 2]])
    dataset = generate_dataset(key, cov_matrix, num_samples=1_000_000)
    kde = MultiVariateGaussianKDE.from_covariance(dataset, covariance=cov_matrix)
    return kde, cov_matrix


def test_covariance_with_per_dimension_bandwidth(kde_fixture):
    kde, cov_matrix = kde_fixture
    dataset = kde.dataset

    # Test that the covariance of the KDE is close to the generating matrix
    cov_matrix = jnp.array([[1, 0.1], [0.1, 1]])
    kde = MultiVariateGaussianKDE.from_covariance(dataset, cov_matrix)
    assert jnp.allclose(kde.covariance, cov_matrix, atol=1e-1)

    # Test that the covariance of the KDE is close to the generating matrix with
    # different bandwidths
    cov_matrix = jnp.array([[3, 1], [10, 2]])
    kde = MultiVariateGaussianKDE.from_covariance(dataset, cov_matrix)
    assert jnp.allclose(kde.covariance, cov_matrix, atol=1e-1)


def test_evaluate_method(kde_fixture):
    kde, _ = kde_fixture

    # Generate points for evaluation
    points = jnp.linspace(-3, 3, 100).reshape(2, -1)

    # Test the evaluate method
    result = kde.evaluate(points)
    assert np.all(result >= 0)
    assert np.all(np.isfinite(result))


def test_call_method(kde_fixture):
    kde, _ = kde_fixture

    # Generate points for evaluation
    points = jnp.linspace(-3, 3, 100).reshape(2, -1)

    # Test the __call__ method
    result_call = kde(points)
    assert np.all(result_call >= 0)
    assert np.all(np.isfinite(result_call))


def test_pdf_method(kde_fixture):
    kde, _ = kde_fixture

    # Generate points for evaluation
    points = jnp.linspace(-3, 3, 100).reshape(2, -1)

    # Test the pdf method
    result_pdf = kde.pdf(points)
    assert np.all(result_pdf >= 0)
    assert np.all(np.isfinite(result_pdf))


def test_logpdf_method(kde_fixture):
    kde, _ = kde_fixture

    # Generate points for evaluation
    points = jnp.linspace(-3, 3, 100).reshape(2, -1)

    # Test the logpdf method
    result_logpdf = kde.logpdf(points)
    assert np.all(np.isfinite(result_logpdf))
