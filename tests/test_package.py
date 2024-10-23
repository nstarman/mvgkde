"""Test the package itself."""

import importlib.metadata

import mvgkde


def test_version():
    assert importlib.metadata.version("mvgkde") == mvgkde.__version__
