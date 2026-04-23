# VINFTRJP

## Overview

This repository contains the official implementation and experimental code for the paper "A Novel Framework using Variational Inference with Normalizing Flows to Train Transport Reversible Jump Proposals". The project provides a complete framework for reproducing all experiments presented in the paper.

## Quick Start

### Prerequisites

- Python 3.12
- [UV](https://docs.astral.sh/uv/) package manager

### Installation

1. Clone the repository:

    ```bash
    git clone git@github.com:Palantir-zoe/vinftrjp.git
    cd vinftrjp
    ```

2. Install dependencies using [UV](https://docs.astral.sh/uv/):

    ```bash
    uv sync
    ```

### Running Experiments

To execute all experiments with 10 independent runs:

```bash
uv run scripts/run.py --end=10
uv run scripts/plot.py --end=10
```

This will run experiments from run 1 to run 10 inclusively.

### Change-Point Example

To run the coal-mining disasters change-point example with the collapsed conditional shared CTP proposal:

```bash
uv run -m scripts.change_point.run_change_point_rjmcmc --samples 40000 --k-max 10
```

The default implementation partially collapses the segment intensities, trains a single conditional shared flow on the saturated change-point space, and then runs RJMCMC with exact collapsed RJ correction. To compare against independent model-specific collapsed flows, add `--independent-flows`.

## Project Structure

The first-level folder structure of this library VINFTRJP is presented below:

- data/: Datasets and processed data
    - raw/: Contains original, immutable datasets. Files here should never be modified.
- docs/: Experiment outputs
    - figures/: All plots, charts, and visualizations generated during experiments.
- scripts/: High-level execution scripts
    - run.py: Main experiment runner
    - plot.py: Main experment plotter
- src/: Source code  (Core Implementation)
- .gitignore: Git ignore patterns
- .python-version: Python version specification
- LICENSE: Open-source license
- pyproject.toml: Project dependencies and metadata
- README.md: Basic information and setup guide
- uv.lock: Locked project dependencies
