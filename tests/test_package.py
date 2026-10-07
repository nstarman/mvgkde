# Copyright (c) 2025 Nathaniel Starkman
# SPDX-License-Identifier: Apache-2.0

"""Test the package itself."""

import importlib.metadata

import mvgkde


def test_version():
    assert importlib.metadata.version("mvgkde") == mvgkde.__version__
