# /// script
# requires-python = ">=3.11"
# dependencies = ["contourpy", "numpy", "resvg-py"]
# ///
"""Draw the mvgkde logo: a kernel density estimate, as filled contours.

Five data points, each with its own tilted Gaussian kernel, summed into a
density and drawn as filled contours in the colours of the README's plots, on a
rounded square. The logo is written as an SVG, sharp at any size; for a bitmap,
name a .png and give its size::

    uv run docs/_static/make_logo.py                     # favicon.svg
    uv run docs/_static/make_logo.py --size 2048 big.png
"""

import argparse
import itertools
from pathlib import Path

import contourpy
import numpy as np

# The data points and each one's kernel covariance, in the square [-1.5, 1.5]^2.
POINTS = np.array([[-0.9, -0.5], [0.1, 0.4], [0.8, -0.3], [-0.2, -0.9], [0.5, 0.9]])
COVARIANCES = np.array(
    [
        [[0.30, 0.18], [0.18, 0.20]],
        [[0.18, -0.10], [-0.10, 0.30]],
        [[0.25, 0.12], [0.12, 0.14]],
        [[0.20, 0.00], [0.00, 0.10]],
        [[0.12, 0.08], [0.08, 0.22]],
    ],
)
FLOOR = 0.08  # the lowest contour; below it the square shows through
# The bands' colours, lowest first: matplotlib's gist_earth_r, as the README's
# plots use, sampled at each band's middle.
COLOURS = ["#e7c9bf", "#c0a565", "#9eb059", "#599f4a", "#388a6a", "#24647c", "#0c1976"]
BACKGROUND = "#fbf7f4"
GRID = 300  # points a side the density is evaluated on
TOLERANCE = 0.002  # how far a simplified outline may stray, of the square's 3

SVG = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="-1.5 -1.5 3 3"
  width="512" height="512">
  <clipPath id="square">
    <rect x="-1.5" y="-1.5" width="3" height="3" rx="0.35"/>
  </clipPath>
  <rect x="-1.5" y="-1.5" width="3" height="3" rx="0.35" fill="{background}"/>
  <g clip-path="url(#square)" fill-rule="evenodd">
{bands}
  </g>
</svg>
"""


def density(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the sum of the normalised Gaussian kernels at ``x``, ``y``."""
    z = np.zeros_like(x)
    for mu, cov in zip(POINTS, COVARIANCES, strict=True):
        d = np.stack([x - mu[0], y - mu[1]], axis=-1)
        mahalanobis = np.einsum("...i,ij,...j", d, np.linalg.inv(cov), d)
        z += np.exp(-0.5 * mahalanobis) / np.sqrt(np.linalg.det(cov))
    return z


def simplify(points: np.ndarray) -> np.ndarray:
    """Return ``points`` with those within TOLERANCE of a straight run dropped."""
    a, b = points[0], points[-1]
    ab = b - a
    q = points - a
    span = np.hypot(*ab)
    if span:
        dist = np.abs(ab[0] * q[:, 1] - ab[1] * q[:, 0]) / span
    else:
        dist = np.hypot(q[:, 0], q[:, 1])
    i = int(np.argmax(dist))
    if dist[i] <= TOLERANCE:
        return np.array([a, b])
    return np.vstack([simplify(points[: i + 1])[:-1], simplify(points[i:])])


def ring(points: np.ndarray) -> str:
    """Return a closed outline as a smooth SVG path: Catmull-Rom, as Beziers."""
    p = simplify(points)[:-1]  # a ring repeats its first point at its end
    p0, p2, p3 = np.roll(p, 1, 0), np.roll(p, -1, 0), np.roll(p, -2, 0)
    c1, c2 = p + (p2 - p0) / 6, p2 - (p3 - p) / 6

    def xy(v: np.ndarray) -> str:
        return f"{v[0]:.3f} {-v[1]:.3f}"  # SVG's y runs down

    spans = (f"C{xy(a)} {xy(b)} {xy(c)}" for a, b, c in zip(c1, c2, p2, strict=True))
    return f"M{xy(p[0])}" + "".join(spans) + "Z"


def svg() -> str:
    """Return the logo as SVG text."""
    # Past the square, so the contours close outside it and are clipped to it.
    x, y = np.mgrid[-1.7 : 1.7 : GRID * 1j, -1.7 : 1.7 : GRID * 1j]
    z = density(x, y)
    levels = np.linspace(FLOOR, z.max(), len(COLOURS) + 1)[:-1]
    tracer = contourpy.contour_generator(x, y, z, fill_type="OuterOffset")
    bands = []
    # Each band is everything above its level, painted lowest first, so a band
    # lies under the ones above it and no seam shows where they meet.
    for colour, level in zip(COLOURS, levels, strict=True):
        polygons, offsets = tracer.filled(level, np.inf)
        d = "".join(
            ring(points[start:stop])
            for points, offset in zip(polygons, offsets, strict=True)
            for start, stop in itertools.pairwise(offset)
        )
        bands.append(f'    <path fill="{colour}" d="{d}"/>')
    return SVG.format(background=BACKGROUND, bands="\n".join(bands))


def main() -> None:
    """Parse the command line and save the logo."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "out",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("favicon.svg"),
        help="output file, SVG or PNG by its extension (default: favicon.svg)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="pixels per side, for a PNG",
    )
    args = parser.parse_args()

    if args.out.suffix == ".svg":
        args.out.write_text(svg())
    else:
        import resvg_py  # noqa: PLC0415  # only a PNG needs a renderer

        png = resvg_py.svg_to_bytes(svg_string=svg(), width=args.size)
        args.out.write_bytes(bytes(png))


if __name__ == "__main__":
    main()
