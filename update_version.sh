#!/bin/bash
# Script to update version numbers in the library

set -e

# Check if a version number is provided
if [ $# -ne 1 ]; then
  echo "Usage: $0 <new_version>"
  echo "Example: $0 0.2.0"
  exit 1
fi

NEW_VERSION=$1

# Update version in _version.py
echo "Updating version to $NEW_VERSION in _version.py"
cat > src/ml_library/_version.py << EOF
"""Version information."""

__version__ = "$NEW_VERSION"
EOF

# Update version in pyproject.toml
echo "Updating version in pyproject.toml"
# Using poetry version command if poetry is installed
if command -v poetry &> /dev/null; then
  poetry version $NEW_VERSION
else
  # Fallback to sed if poetry is not installed
  sed -i '' "s/^version = \"[0-9]*\.[0-9]*\.[0-9]*\"/version = \"$NEW_VERSION\"/g" pyproject.toml
fi

# Update CHANGELOG.md
echo "Don't forget to update CHANGELOG.md with the new version details!"
echo "Add a new section ## [$NEW_VERSION] - $(date +%Y-%m-%d)"

echo "Version updated to $NEW_VERSION"
echo "Run tests and verify all files before committing!"
