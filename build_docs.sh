#!/bin/zsh
# Build the ML Library documentation

# Set error handling
set -e

# Change to project root directory
cd "$(dirname "$0")"

# Check for Sphinx
if ! command -v sphinx-build &> /dev/null; then
    echo "Sphinx not found. Installing dependencies..."
    pip install -r requirements-dev.txt
fi

# Build the documentation
echo "Building documentation..."
cd docs
make html

# Open the documentation in the browser (macOS specific)
echo "Documentation built successfully!"
echo "Opening documentation in browser..."
open build/html/index.html

echo "You can also access the documentation at: file://$(pwd)/build/html/index.html"
