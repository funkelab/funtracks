# Developer Guide

## Managing dependencies with `uv`
This repo has been set up to use `uv` for developer dependency management.
Install `uv` according to [their instructions](https://docs.astral.sh/uv/getting-started/installation/).

Dev dependencies are specified in the `dependency-groups` section of the `pyproject.toml`.
The `dev` group is installed by default with `uv`, so running `uv run ...` should allow
you to use all the developer tools specified in this section.

Dependency groups are different from the extra dependencies specified in
`project.optional-dependencies`, so they cannot be installed with `pip install .[dev]`
and are not packaged and distributed with the library.

## Running tasks with `just`
For convenience, common development tasks like building docs or running test with coverage
can be run with `just`. Install `just` according to [their instructions](https://just.systems/man/en/packages.html).

Tasks are defined in the `justfile`. These tasks are just shorthand for common commands that
you would otherwise type into your shell over and over.

## Updating and previewing documentation
We are using mkdocs-material and mike to get versioned documentation. Mike will
push changes to the gh-pages branch, which we can serve from the Github Pages settings.
This should happen automatically from the github action upon push to 'main' (with alias 'dev') and newly tagged version (with alias 'latest'),
but we are documenting it here in case we need to do it manually at some point.

To publish a commit to the gh-pages branch manually:
```bash
uv run mike deploy <version> <alias>
```
or with just tasks:
```bash
just docs-deploy <version> <alias>
```

To preview pushed changes locally:
```bash
uv run mike serve
```
or with just tasks:
```bash
just docs-serve
```
Remember not to push your local gh-pages branch on accident after previewing changes.
To reset your local branch to the remote gh-pages, you can run:
```bash
git fetch
git checkout gh-pages
git reset --hard origin/gh-pages
```
