"""Tests comparing mvgkde to jax.scipy."""

import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose
from scipy.stats import gaussian_kde as scipy_gaussian_kde

from .test_vs_jax_kde import bw_method_strategy, dataset_strategy
from mvgkde import gaussian_kde


@st.composite
def dataset_marginal_strategy(draw, dataset_strat=dataset_strategy):
    # Draw the dataset from the provided strategy
    dataset = draw(dataset_strat)

    # Get the length of the dataset
    max_length = dataset.shape[0]

    # Draw a random length for the integer array, between 1 and max_length
    length = draw(st.integers(min_value=1, max_value=max_length))

    # Generate a unique integer array with values between 0 and max_length - 1
    marginal_indices = draw(
        arrays(
            dtype=np.int32,
            shape=length,
            elements=st.integers(min_value=0, max_value=max_length - 1),
            unique=True,
        ),
    )

    # Return both the dataset and the integer array
    return dataset, marginal_indices


# =============================================================================
# Test the hypothesis strategies


@settings(max_examples=50)
@given(dataset_marginal=dataset_marginal_strategy())
def test_marginal_strategy(dataset_marginal):
    dataset, marginal_indices = dataset_marginal
    assert len(marginal_indices) >= 1
    assert len(marginal_indices) <= dataset.shape[0]
    assert len(marginal_indices) == len(set(marginal_indices))


##############################################################################
# Test the mvgkde vs scipy implementation


@settings(deadline=None)  # TODO: address jit compilation speed change
@given(dataset_marginals=dataset_marginal_strategy(), bw=bw_method_strategy)
def test_marginal(dataset_marginals, bw):
    # Unpack the dataset and marginal indices
    dataset, marginal_indices = dataset_marginals

    # Make KDE objects from the dataset
    kde = gaussian_kde(dataset, bw_method=bw)
    scipy_kde = scipy_gaussian_kde(dataset, bw_method=bw)

    assert_allclose(kde.dataset, scipy_kde.dataset)
    assert_allclose(kde.weights, scipy_kde.weights)
    assert_allclose(kde.neff, scipy_kde.neff)
    assert_allclose(kde.covariance, scipy_kde.covariance, rtol=1e-3)
    assert_allclose(kde.inv_cov, scipy_kde.inv_cov, rtol=1e-3)

    # Create Marginal KDE objects
    mkde = kde.marginal(marginal_indices)
    scipy_mkde = scipy_kde.marginal(marginal_indices)

    # Check that the marginal KDEs have the same attributes
    assert_allclose(mkde.dataset, scipy_mkde.dataset)
    assert_allclose(mkde.weights, scipy_mkde.weights)
    assert_allclose(mkde.neff, scipy_mkde.neff)
    assert_allclose(mkde.covariance, scipy_mkde.covariance, rtol=1e-3)
    assert_allclose(mkde.inv_cov, scipy_mkde.inv_cov, rtol=1e-3)
