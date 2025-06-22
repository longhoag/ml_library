"""Setup script for ml_library package."""

import os
import re

from setuptools import find_packages, setup

# Read the contents of README.md
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as readme_file:
    long_description = readme_file.read()


# Get version from _version.py file
def get_version() -> str:
    """Get version number from _version.py file.

    Returns
    -------
    str
        The version number string.
    """
    version_file = os.path.join(this_directory, "src", "ml_library", "_version.py")
    with open(version_file, "r", encoding="utf-8") as version_file_handle:
        version_match = re.search(
            r"^__version__ = ['\"]([^'\"]*)['\"]", version_file_handle.read(), re.M
        )
        if version_match:
            return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# Package meta-data
NAME = "ml_library"
DESCRIPTION = "A production-ready machine learning library"
URL = "https://github.com/longhoag/ml_library"
EMAIL = "longhhoang1202@gmail.com"
AUTHOR = "Long Hoang"
REQUIRES_PYTHON = ">=3.8.0"
VERSION = get_version()

# Required packages
REQUIRED = [
    "numpy>=1.20.0",
    "pandas>=1.3.0",
    "scikit-learn>=1.0.0",
    "matplotlib>=3.4.0",
    "joblib>=1.0.0",
]

# Optional packages
EXTRAS = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=3.0.0",
        "flake8>=4.0.0",
        "black>=22.0.0",
        "isort>=5.10.0",
        "mypy>=0.9.0",
        "sphinx>=4.4.0",
        "sphinx-rtd-theme>=1.0.0",
        "twine>=4.0.0",
        "build>=0.10.0",
    ],
    "tensorflow": ["tensorflow>=2.8.0"],
    "torch": ["torch>=1.10.0"],
}

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Operating System :: OS Independent",
    ],
    keywords="machine learning, ML, data science, sklearn, scikit-learn",
    project_urls={
        "Documentation": "https://ml-library.readthedocs.io",
        "Source": "https://github.com/longhoag/ml_library",
        "Tracker": "https://github.com/longhoag/ml_library/issues",
    },
    entry_points={
        "console_scripts": [
            # Add command line scripts here if needed
        ],
    },
)
