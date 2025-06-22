"""Setup script for ml_library package."""

import os
from setuptools import setup, find_packages


# Read the contents of README.md
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Package meta-data
NAME = 'ml_library'
DESCRIPTION = 'A production-ready machine learning library'
URL = 'https://github.com/username/ml_library'
EMAIL = 'your.email@example.com'
AUTHOR = 'Your Name'
REQUIRES_PYTHON = '>=3.8.0'

# Required packages
REQUIRED = [
    'numpy>=1.20.0',
    'pandas>=1.3.0',
    'scikit-learn>=1.0.0',
    'matplotlib>=3.4.0',
    'joblib>=1.0.0',
]

# Optional packages
EXTRAS = {
    'dev': [
        'pytest>=7.0.0',
        'pytest-cov>=3.0.0',
        'flake8>=4.0.0',
        'black>=22.0.0',
        'isort>=5.10.0',
        'mypy>=0.9.0',
        'sphinx>=4.4.0',
        'sphinx-rtd-theme>=1.0.0',
    ],
    'tensorflow': ['tensorflow>=2.8.0'],
    'torch': ['torch>=1.10.0'],
}

setup(
    name=NAME,
    version='0.1.0',
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
