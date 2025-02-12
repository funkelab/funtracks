# Funtracks
A data model for cell tracking with actions, undo history, persistence, and more!

Designed for use with the [finn tracks viewer](https://github.com/funkelab/finn) and [motile tracker](https://github.com/funkelab/motile_tracker). Provides a data model for programmatic editing of tracks that can be used by a UI like finn or by a machine agent for active learning.

## Installation
`pip install funtracks`

Or use pixi!

## Roadmap
Features already included in funtracks:
- fully undo-able action history

Features that will be included in funtracks:
- both in-memory and out-of-memory tracks
  - in-memory using networkx (slower, pure python) or spatial_graph (faster, compiled C) data structures
  - out-of-memory using zarr for segmentations and SQLite/PostGreSQL for graphs
- functions to import from and export to common file structures (csv, segmentation relabeled by track id)
- generic features with the option to automatically update features on tracks editing actions
