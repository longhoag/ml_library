# Migration from Pip to Poetry

This document describes the migration from using pip/setuptools to Poetry for package management in this project.

## What is Poetry?

[Poetry](https://python-poetry.org/) is a modern dependency management and packaging tool for Python. It simplifies dependency management and packaging by providing:

- Lock file for reproducible builds
- Better dependency resolution
- Simplified packaging commands
- Integrated virtual environment management
- Easier publishing to PyPI

## Changes Made

The migration involved the following changes:

1. **Configuration**: Moving dependencies and build configurations from `setup.py`, `setup.cfg`, and requirements files into a single `pyproject.toml` file.

2. **Dependency Management**: Using `poetry.lock` to lock dependencies for reproducible builds.

3. **Scripts**: Updated scripts to use Poetry for building and publishing.

4. **CI/CD**: Updated GitHub Actions workflows to use Poetry.

5. **Documentation**: Updated README and created CONTRIBUTING.md with Poetry instructions.

## Key Benefits

- **Simplified Dependencies**: All dependencies are specified in a single location.
- **Reproducible Builds**: Poetry's lock file ensures everyone gets the same dependencies.
- **Dev Environment**: Easy creation of isolated development environments.
- **Build and Publish**: Streamlined commands for packaging and publishing.

## Getting Started with Poetry

See the [CONTRIBUTING.md](CONTRIBUTING.md) file for instructions on setting up a development environment with Poetry.

## Legacy Files

The following files have been kept for reference but are no longer used:

- `requirements.txt`
- `requirements-dev.txt`
- `setup.py`
- `setup.cfg`

These files can be removed in the future when the migration is complete and stable.
