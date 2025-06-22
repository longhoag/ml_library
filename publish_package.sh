#!/bin/bash
# Script to build and publish the package to PyPI

set -e

# Check arguments
if [ "$1" != "--production" ] && [ "$1" != "--test" ]; then
  echo "Usage: $0 --test|--production"
  echo "  --test: Upload to Test PyPI"
  echo "  --production: Upload to Production PyPI"
  exit 1
fi

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/

# Build the package
echo "Building package..."
python -m pip install --upgrade build
python -m build

# Upload to PyPI
if [ "$1" == "--test" ]; then
  echo "Uploading to Test PyPI..."
  python -m pip install --upgrade twine
  python -m twine upload --repository testpypi dist/*
elif [ "$1" == "--production" ]; then
  echo "Are you sure you want to upload to Production PyPI? (y/n)"
  read -r confirmation
  if [ "$confirmation" != "y" ]; then
    echo "Aborted."
    exit 0
  fi

  echo "Uploading to Production PyPI..."
  python -m pip install --upgrade twine
  python -m twine upload dist/*
fi

echo "Done!"
