"""MultiVariate Gaussian Kernel Density Estimator (mvgkde)."""

__all__: list[str] = [
    "__version__",
    "MultiVariateGaussianKDE",
    "gaussian_kde",
]

from ._core import MultiVariateGaussianKDE, gaussian_kde
from ._version import version as __version__
