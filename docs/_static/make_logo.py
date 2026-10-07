# /// script
# requires-python = ">=3.11"
# dependencies = ["contourpy", "numpy", "resvg-py"]
# ///
# Copyright (c) 2025 Nathaniel Starkman
# SPDX-License-Identifier: Apache-2.0

"""Draw the mvgkde logo: a kernel density estimate of a handwritten M.

Points scattered along a slanted script M, from a fixed seed, and their kernel
density estimate with one full bandwidth matrix, tilted with the slant, drawn
as filled contours in the colours of the README's plots on a rounded square.
The logo is written as an SVG, sharp at any size; for a bitmap, name a .png and
give its size::

    uv run docs/_static/make_logo.py                     # favicon.svg
    uv run docs/_static/make_logo.py --size 2048 big.png
"""

import argparse
import itertools
from pathlib import Path

import contourpy
import numpy as np

# A script M's stroke, upright: a lead-in curling up the left leg, an arch down
# into the middle, a second arch, and the right leg ending in a flick. It is
# slanted, scaled into the square [-1.5, 1.5]^2, and points scattered along it.
STROKE = np.array(
    [
        [-1.05, -0.75],
        [-0.95, -0.2],
        [-0.82, 0.55],
        [-0.62, 0.85],
        [-0.42, 0.6],
        [-0.25, 0.0],
        [-0.12, -0.45],
        [0.0, -0.1],
        [0.15, 0.55],
        [0.38, 0.85],
        [0.58, 0.55],
        [0.68, 0.0],
        [0.72, -0.6],
        [0.85, -0.85],
        [1.05, -0.7],
    ],
)
SLANT = 0.32  # x moves right by this much per unit up
SCALE = 0.88
SEED, SAMPLES = 2, 60  # the draw: which, and how many points
SCATTER = 0.06  # the points' spread about the stroke
BANDWIDTH = 0.10  # the kernels' width across the slant; 1.6 times it along it
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


def curve(points: np.ndarray, per_span: int = 60) -> np.ndarray:
    """Return points along an open Catmull-Rom curve through ``points``."""
    p = np.vstack([points[:1], points, points[-1:]])
    t = np.linspace(0, 1, per_span, endpoint=False)[:, None]
    spans = []
    for p0, p1, p2, p3 in zip(p, p[1:], p[2:], p[3:], strict=False):
        spans.append(
            0.5
            * (
                2 * p1
                + (p2 - p0) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
                + (3 * p1 - p0 - 3 * p2 + p3) * t**3
            ),
        )
    return np.concatenate(spans)


def sample() -> np.ndarray:
    """Return the data: SAMPLES points scattered along the slanted stroke."""
    line = curve(STROKE)
    line[:, 0] += SLANT * line[:, 1]
    line *= SCALE
    # Spread evenly by length along the stroke, then scattered about it.
    length = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(line, axis=0).T))])
    rng = np.random.default_rng(SEED)
    at = np.sort(rng.uniform(0, length[-1], SAMPLES))
    points = np.column_stack(
        [np.interp(at, length, line[:, 0]), np.interp(at, length, line[:, 1])],
    )
    return points + rng.normal(0, SCATTER, points.shape)


def density(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the kernel density estimate of the data at ``x``, ``y``.

    Every kernel shares one full bandwidth matrix, stretched along the slant.
    """
    along = np.array([SLANT, 1.0]) / np.hypot(SLANT, 1.0)
    across = np.array([-along[1], along[0]])
    bandwidth = (1.6 * BANDWIDTH) ** 2 * np.outer(along, along)
    bandwidth += BANDWIDTH**2 * np.outer(across, across)
    inverse = np.linalg.inv(bandwidth)
    norm = 1 / np.sqrt(np.linalg.det(bandwidth))  # FLOOR is on this scale
    z = np.zeros_like(x)
    for mu in sample():
        d = np.stack([x - mu[0], y - mu[1]], axis=-1)
        z += norm * np.exp(-0.5 * np.einsum("...i,ij,...j", d, inverse, d))
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
