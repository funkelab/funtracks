# Contributing to funtracks

## Setting up a development environment

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and
[just](https://just.systems) as a task runner. Install both first:

- [uv installation](https://docs.astral.sh/uv/getting-started/installation/)
- [just installation](https://just.systems/man/en/packages.html)

`just` is only a convenience wrapper — every recipe is a one-line `uv run ...`
command you can copy out of the `justfile` and run directly.

```bash
git clone https://github.com/funkelab/funtracks
cd funtracks
just install
```

`just install` creates the virtual environment (`uv sync`) and installs the
pre-commit hooks. Run `just` to see all available recipes.

## Dependencies

Dependencies are declared in `pyproject.toml` and have the following conventions.

### Every dependency needs a lower and an upper bound

Give every requirement — runtime, `test`, `docs` and `dev` alike — both bounds,
with the upper bound at the **next major version**:

```toml
dependencies = [
    "numpy>=1.26,<3",      # yes
    "numpy>=1.26",         # no: unbounded above
    "numpy",               # no: unbounded in both directions
    "numpy==2.3.1",        # no: that is the lock file's job
]
```

Note that a bound spans majors whenever the older ones still work — `<3` is the
next major above the *newest* release you support, not a companion to the floor.

The upper bound excludes the next major release, since under semantic versioning
that is where the project is permitted to break you.

- **Without an upper bound**, a major release upstream reaches users through a
  fresh resolve of `pyproject.toml` — a new contributor's `uv sync`, or anyone
  installing this package as a dependency — and breaks them before anyone here
  has run the tests against it.
- **Without a lower bound**, the resolver is free to pick something ancient to
  satisfy an unrelated conflict, and the failure surfaces as a missing attribute
  rather than a resolution error.
- **Pinning exactly** (`==`) belongs in `uv.lock`, not here. Exact pins in
  `pyproject.toml` make this package unusable alongside anything with a slightly
  different pin.

#### Choosing the lower bound

The lower bound is the **oldest version that still works**, and how hard you
should work to keep it low depends on who has to live with it.

For **runtime `dependencies`**, keep the floor as low as you can stand. These are
installed into someone else's environment, next to other packages with their own
constraints on the same libraries. A floor of "whatever was current the day I
typed `uv add`" is an arbitrary co-installation hazard: it excludes anyone whose
environment is held slightly behind by an unrelated package, and the resolver
reports that as an unsatisfiable conflict rather than anything actionable. Set it
to the oldest release providing the APIs you actually use — typically the version
that introduced the newest function or argument you call — not to today's
release.

For **`test`, `docs` and `dev` groups**, the floor costs nothing. Nothing else
shares that environment; it is created from this project's own lock file. Use the
current release when you add the tool and move on.

The upper bound rule is the same everywhere.

For pre-1.0 dependencies (`0.x`), a minor bump plays the role of a major, so the
upper bound goes at the next minor: `ruff>=0.16,<0.17`.

Widening an upper bound is a deliberate, reviewed change: edit the constraint,
re-lock, and let CI tell you whether the new major actually breaks anything. The
`Outdated dependencies` workflow described below exists to surface the releases
that these bounds are holding back.

Because versions are pinned, new upstream releases are *not* picked up
automatically. Upgrading is a deliberate step:

```bash
just upgrade                  # upgrade everything within the declared constraints
just upgrade-package numpy    # upgrade a single dependency
```

Commit the resulting `uv.lock` alongside the change. Outside of these commands
the lock file stays put — routine `just test` / `uv run` only read it.

These commands only move versions *within* the constraints in `pyproject.toml`;
they never rewrite the constraints themselves. A dependency declared as
`foo>=1,<2` will stay on 1.x however often you upgrade. Moving to 2.x means
editing the bound in `pyproject.toml` by hand, then re-locking — which is the
point at which you find out whether the new major breaks anything.

Dependabot opens a monthly pull request bumping `uv.lock`, and CI runs the full
test matrix against the new versions, so most upgrades arrive as a PR you only
need to review and merge.

Releases that fall *outside* the declared bounds never reach those PRs, so a
monthly `Outdated dependencies` workflow lists them in a GitHub issue instead.
It only reports; acting on it is the manual bound-widening step above.

## Module layout

Implementation modules are **underscore-prefixed** (`_example.py`, `_io.py`), and
everything users are meant to touch is re-exported from `__init__.py` and listed
in `__all__`:

```
src/funtracks/
    __init__.py    # from ._example import greet;  __all__ = [..., "greet"]
    _example.py    # implementation
```

```python
from funtracks import greet  # yes
from funtracks._example import greet  # works, but not the supported path
```

This is mainly about the generated API docs. `mkdocs-api-autonav` builds a page
per public module, and `mkdocstrings` is configured with `filters: ["!^_"]`, so
a public `example.py` alongside a re-export in `__init__.py` documents `greet`
twice — once on its own module page, once on the package page — and modules with
nothing public left to show still get an empty nav entry. Underscore-prefixing
the module removes it from the nav, leaving one page generated from `__all__`.

The convention pays off twice over:

- **The public API is a decision, not a side effect of the file layout.** Adding
  a name to `__all__` is deliberate; splitting `_run.py` into `_run.py` and
  `_dtype.py` is then a refactor users never see.
- **Docs match the import path you actually support.** Anything documented is
  importable from the top-level package.

Nothing enforces this mechanically — if a package genuinely warrants a public
submodule (a `cli`, or a heavy optional subpackage users import directly), name
it without the underscore and let it have its own docs page.

## Tests, linting and type checking

```bash
just test        # run the test suite
just test-cov    # run the test suite with a coverage report
just lint        # run ruff (lint + format) and the other pre-commit hooks
just typecheck   # run mypy via pre-commit
```

Linting, formatting and type checking are all handled via
[pre-commit](https://pre-commit.com/) hooks (ruff for lint/format, mypy for type
checking), so most style issues are fixed automatically when you commit, and
mypy always runs with the same stub/type-dependency setup declared in
`.pre-commit-config.yaml` whether you run it locally or it runs in CI.

## Documentation

Documentation is built with [mkdocs-material](https://squidfunk.github.io/mkdocs-material/)
and API pages are generated from docstrings (Google style).

```bash
just docs-serve  # preview the docs locally with live reload
just docs-build  # build the docs in strict mode, as CI does
```

We use mike to get versioned documentation. Mike pushes changes to the
`gh-pages` branch, which is served from the GitHub Pages settings. This
happens automatically via CI on push to `main` (alias `dev`) and on a newly
tagged version (alias `latest`), but is documented here in case it needs to be
done manually:

```bash
just docs-deploy <version> <alias>   # or: uv run mike deploy <version> <alias>
just docs-serve                      # preview pushed changes: uv run mike serve
```

Remember not to push your local `gh-pages` branch by accident after previewing
changes. To reset your local branch to match the remote:

```bash
git fetch
git checkout gh-pages
git reset --hard origin/gh-pages
```

## Pull requests

CI runs the test suite across Linux, macOS and Windows
and builds the docs. Type checking and linting run via pre-commit
(locally on commit, and in CI through pre-commit.ci) rather than as a separate
CI job. Please make sure `just test`, `just lint` and `just typecheck` pass
locally before opening a pull request.

## Releasing

Releases are triggered by pushing a version tag. `setuptools-scm` derives the
version from the tag, and CI builds and publishes to PyPI.

```bash
just release v0.1.0
```
Publishing uses [trusted publishing](https://docs.pypi.org/trusted-publishers/),
which must be configured once on PyPI for this project.
