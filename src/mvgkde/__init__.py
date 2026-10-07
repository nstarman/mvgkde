# Copyright (c) 2025 Nathaniel Starkman
# SPDX-License-Identifier: Apache-2.0

"""MultiVariate Gaussian Kernel Density Estimator (mvgkde)."""

__all__: list[str] = [
    "MultiVariateGaussianKDE",
    "__version__",
    "gaussian_kde",
]

from ._core import MultiVariateGaussianKDE, gaussian_kde
from ._version import version as __version__
