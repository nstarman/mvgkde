#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["nox", "nox_uv"]
# ///
"""Nox configuration file."""

import shutil
from pathlib import Path

import nox
from nox_uv import session

DIR = Path(__file__).parent.resolve()

nox.needs_version = ">=2024.3.2"
nox.options.default_venv_backend = "uv"


@session(uv_groups=["lint"], reuse_venv=True, default=True)
def lint(s: nox.Session) -> None:
    """Run the linter."""
    s.run(
        "pre-commit",
        "run",
        "--all-files",
        "--show-diff-on-failure",
        *s.posargs,
    )


@session(uv_groups=["lint"], reuse_venv=True, default=True)
def pylint(s: nox.Session) -> None:
    """Run PyLint."""
    # This needs to be installed into the package environment, and is slower
    # than a pre-commit check
    s.install(".")
    s.run("pylint", "mvgkde", *s.posargs)


@session(uv_groups=["test"], reuse_venv=True, default=True)
def tests(s: nox.Session) -> None:
    """Run the unit and regular tests."""
    # Install the project
    s.install(".")
    s.run("pytest", *s.posargs)


@session(reuse_venv=True)
def build(s: nox.Session) -> None:
    """Build an SDist and wheel."""
    build_path = DIR.joinpath("build")
    if build_path.exists():
        shutil.rmtree(build_path)

    s.install("build")
    s.run("python", "-m", "build")


if __name__ == "__main__":
    nox.main()
