# Distribution and Deployment Guide

This document provides instructions for packaging, versioning, and distributing the ml_library project.

## Versioning

The project follows [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality in a backward-compatible manner
- **PATCH**: Backward-compatible bug fixes

### Updating the Version

Use the provided script to update the version across the project:

```bash
./update_version.sh X.Y.Z
```

Then update the CHANGELOG.md file with details about the new version.

## Building the Package

To build the package:

```bash
# Install build dependencies
pip install --upgrade build

# Build the package (creates both wheel and source distribution)
python -m build
```

This will create distribution files in the `dist/` directory.

## Testing Distribution Locally

You can install the package locally to test:

```bash
pip install -e .
```

Or install the built distribution:

```bash
pip install dist/ml_library-X.Y.Z-py3-none-any.whl
```

## Publishing to PyPI

### Test PyPI

Before releasing to the main PyPI repository, you can test with Test PyPI:

```bash
./publish_package.sh --test
```

Then install from Test PyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ml_library
```

### Production PyPI

When you're ready to publish to the main PyPI repository:

```bash
./publish_package.sh --production
```

## Release Checklist

Before releasing a new version:

1. Ensure all tests pass (`pytest`)
2. Check test coverage is at least 90% (`pytest --cov=ml_library`)
3. Verify documentation builds without errors (`cd docs && make html`)
4. Update version (`./update_version.sh X.Y.Z`)
5. Update CHANGELOG.md with the new version's changes
6. Commit all changes and tag the release (`git tag vX.Y.Z`)
7. Push with tags (`git push --tags`)
8. Build and publish the package (`./publish_package.sh --production`)
9. Create a GitHub release with release notes

## CI/CD Integration

The project includes GitHub Actions workflows that:
1. Run tests on every push
2. Check code quality and test coverage
3. Build documentation

A publishing workflow can be added to automatically deploy new tagged releases to PyPI.
