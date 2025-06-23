# Development Guide

This document outlines the development workflow and contribution guidelines for the ML Library project.

## Development Setup

### Using Poetry (Recommended)

We use [Poetry](https://python-poetry.org/) for dependency management.

1. Install Poetry if you haven't already:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/longhoag/ml_library.git
   cd ml_library
   ```

3. Install dependencies:
   ```bash
   poetry install
   ```

4. Activate the virtual environment:
   ```bash
   poetry shell
   ```

### Common Development Tasks with Poetry

- Add a new dependency:
  ```bash
  poetry add package-name
  ```

- Add a development dependency:
  ```bash
  poetry add --group dev package-name
  ```

- Run tests:
  ```bash
  poetry run pytest
  ```

- Run linting:
  ```bash
  poetry run flake8 src tests
  ```

- Type checking:
  ```bash
  poetry run mypy src
  ```

### Using pip (Legacy)

If you prefer not to use Poetry, you can use pip:

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Code Quality

We use several tools to maintain code quality:

- **Black**: For code formatting
- **Flake8**: For linting
- **isort**: For import sorting
- **mypy**: For type checking
- **pre-commit**: To run these checks automatically before commits

To set up pre-commit:

```bash
poetry run pre-commit install
```

## Versioning

To update the package version:

```bash
./update_version.sh 0.2.1  # Replace with new version number
```

## Publishing

To build and publish the package:

```bash
# For Test PyPI
./publish_package.sh --test

# For Production PyPI
./publish_package.sh --production
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes and add tests
3. Ensure all tests pass and formatting is correct
4. Update documentation if necessary
5. Submit a pull request

## Documentation

The following documentation files are available:

- **README.md**: Main project documentation and overview
- **CHANGELOG.md**: Version history and changes
- **CONTRIBUTING.md**: This file, containing contribution guidelines
- **DISTRIBUTION.md**: Information about building and distributing the package
- **GIT_WORKFLOW.md**: Git workflow guidelines for the project

## Utility Scripts

The repository includes several utility scripts:

- **build_docs.sh**: Script for building the project documentation
- **publish_package.sh**: Script for building and publishing the package to PyPI
- **update_version.sh**: Script for updating the version number across the project

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
