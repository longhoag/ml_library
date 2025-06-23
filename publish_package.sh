#!/bin/bash
# Script to build and publish the package to PyPI using Poetry

set -e

# Check arguments
if [ "$1" != "--production" ] && [ "$1" != "--test" ]; then
  echo "Usage: $0 --test|--production"
  echo "  --test: Upload to Test PyPI"
  echo "  --production: Upload to Production PyPI"
  exit 1
fi

# Check Poetry installation
if ! command -v poetry &> /dev/null; then
  echo "Poetry not found. Please install Poetry first with: pip install poetry"
  exit 1
fi

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf dist/

# Build the package with poetry
echo "Building package..."
poetry build

# Upload to PyPI
if [ "$1" == "--test" ]; then
  echo "Uploading to Test PyPI..."
  poetry config repositories.testpypi https://test.pypi.org/legacy/
  poetry publish -r testpypi
elif [ "$1" == "--production" ]; then
  echo "Are you sure you want to upload to Production PyPI? (y/n)"
  read -r confirmation
  if [ "$confirmation" != "y" ]; then
    echo "Aborted."
    exit 0
  fi

  echo "Uploading to Production PyPI..."
  poetry publish
fi

echo "Done!"
