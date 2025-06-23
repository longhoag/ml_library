#!/bin/zsh
# Build the ML Library documentation

# Set error handling
set -e

# Change to project root directory
cd "$(dirname "$0")"

# Check for Poetry
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Please install Poetry first: https://python-poetry.org/docs/#installation"
    exit 1
fi

# Ensure dev dependencies are installed (including Sphinx)
echo "Ensuring dependencies are installed..."
poetry install --with dev

# Build the documentation
echo "Building documentation..."
cd docs
poetry run make html

# Open the documentation in the browser (macOS specific)
echo "Documentation built successfully!"
echo "Opening documentation in browser..."
open build/html/index.html

echo "You can also access the documentation at: file://$(pwd)/build/html/index.html"
