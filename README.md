# Funtracks
A data model for cell tracking with actions, undo history, persistence, and more!

[![tests](https://github.com/funkelab/funtracks/workflows/tests/badge.svg)](https://github.com/funkelab/funtracks/actions)
[![codecov](https://codecov.io/gh/funkelab/funtracks/branch/main/graph/badge.svg)](https://codecov.io/gh/funkelab/funtracks)

The full documentation can be found [here](https://funkelab.github.io/funtracks/).

----------------------------------

## Installation

pip install funtracks (soon)

## Issues

If you encounter any problems, please
[file an issue](https://github.com/funkelab/funtracks/issues)
along with a detailed description.


# Updating documentation
We are using mkdocs-material and mike to get versioned documentation. Mike will
push changes to the gh-pages branch, which we can serve from the Github Pages settings.This should happen automatically from the github action upon push to main (with alias 'dev') and new tag (with alias 'latest'), but we are documenting it here in case we need to do it manually at some point.

To publish a commit to the gh-pages branch manually:
```bash
uvx --with mkdocs-material --with mike mike deploy <version> <alias>
```
