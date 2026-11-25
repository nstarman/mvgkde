See the [Scientific Python Developer Guide][spc-dev-intro] for a detailed
description of best practices for developing scientific packages.

[spc-dev-intro]: https://learn.scientific-python.org/development/

# Quick development

The fastest way to start with development is to use nox. If you don't have nox,
you can use `pipx run nox` to run it without installing, or `pipx install nox`.
If you don't have pipx (pip for applications), then you can install with
`pip install pipx` (the only case were installing an application with regular
pip is reasonable). If you use macOS, then pipx and nox are both in brew, use
`brew install pipx nox`.

To use, run `nox`. This will lint and test using every installed version of
Python on your system, skipping ones that are not installed. You can also run
specific jobs:

```console
$ nox -s lint  # Lint only
$ nox -s tests  # Python tests
$ nox -s docs -- --serve  # Build and serve the docs
$ nox -s build  # Make an SDist and wheel
```

Nox handles everything for you, including setting up an temporary virtual
environment for each run.

# Setting up a development environment manually

You can set up a development environment by running:

```bash
python3 -m venv .venv
source ./.venv/bin/activate
pip install -v -e .[dev]
```

If you have the
[Python Launcher for Unix](https://github.com/brettcannon/python-launcher), you
can instead do:

```bash
py -m venv .venv
py -m install -v -e .[dev]
```

# Pre-commit

You should prepare pre-commit, which will help you by checking that commits pass
required checks:

```bash
pip install pre-commit # or brew install pre-commit on macOS
pre-commit install # Will install a pre-commit hook into the git repo
```

You can also/alternatively run `pre-commit run` (changes only) or
`pre-commit run --all-files` to check even without installing the hook.

# The code graph

`graphify-out/graph.json` is a
[graphify](https://github.com/Graphify-Labs/graphify) map of this repository:
its symbols and the relationships between them. AI assistants query it instead
of grepping, and you can too. Extraction is local tree-sitter parsing — no model
call, and nothing leaves your machine — and every edge is marked `EXTRACTED`
(explicit in the source) or `INFERRED` (resolved by graphify), so you can tell
what was read from what was deduced.

The graph is committed, and the commit checks rebuild it. Install the pinned
release and register the merge driver once per clone, so two branches that both
changed the graph merge cleanly:

```bash
uv tool install "graphifyy==0.9.119"
PY="$(sed -n '1s/^#!//p' "$(command -v graphify)")"
git config merge.graphify.name "graphify graph.json union merge"
git config merge.graphify.driver "\"$PY\" -m graphify merge-driver %O %A %B"
```

Then query it:

```bash
graphify query "how is the kernel covariance used"  # a scoped subgraph for a question
graphify explain "MultiVariateGaussianKDE"          # one symbol and its neighbours
graphify path "gaussian_kde" "MultiVariateGaussianKDE"  # how two things connect
```

Day to day:

- The rebuild runs with your other commit checks. Like a formatter, it rebuilds
  and stops the commit; `git add graphify-out/graph.json` and commit again.
- Commit without rebuilding: `SKIP=graphify-update git commit ...`.
- If a rebuild is ever refused because the graph would shrink — you deleted code
  — use `GRAPHIFY_FORCE=1 git commit ...`. It usually just shrinks and says
  nothing.
- If the graph ever conflicts in a merge, rebuild rather than edit it:
  `rm graphify-out/graph.json && PYTHONHASHSEED=0 graphify update . --force && git add graphify-out/graph.json`.
- A changed `graph.json` in a pull request is generated output. It is one line,
  collapsed by GitHub, and is not reviewed.
- The rebuild may rewrite the graph the first time you commit in a working
  environment, and the weekly job may rewrite it back. That is expected:
  graphify's output depends on which files are on disk, so a built checkout and
  a fresh clone disagree about community numbering even though the symbols are
  identical.

# Testing

Use pytest to run the unit checks:

```bash
pytest
```

# Coverage

Use pytest-cov to generate coverage reports:

```bash
pytest --cov=mvgkde
```

# Building docs

You can build the docs using:

```bash
nox -s docs
```

You can see a preview with:

```bash
nox -s docs -- --serve
```
