Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
----------

Added
~~~~~
- Future features and improvements will be listed here

[0.2.0] - 2025-06-22
------------------

Added
~~~~~
- Achieved >95% test coverage across all modules
- Expanded test suite with edge case handling
- Comprehensive type annotations supporting mypy
- Enhanced error handling in all preprocessors
- Fixed StandardScaler and MinMaxScaler to match scikit-learn interface
- Improved feature engineering components
- Added example for feature selection workflow
- Expanded logging functionality

Fixed
~~~~~
- StandardPreprocessor initialization and transform handling
- Fixed StandardScaler and MinMaxScaler compatibility with scikit-learn
- Circular import issues in preprocessing modules
- Corrected model evaluation metrics with proper error handling
- All type annotation issues for Python 3.10+ compatibility
- Fixed import/API mismatches throughout the codebase

[0.1.0] - 2023-06-22
------------------

Added
~~~~~
- Initial release with core functionality
- Base Model and Preprocessor abstract classes
- LinearModel, LogisticModel implementation
- RandomForestModel, RandomForestRegressorModel implementation
- StandardPreprocessor, PolynomialPreprocessor implementation
- FeatureSelector for feature selection
- Metrics module with common evaluation metrics
- Visualization module with learning curve plotting
- Comprehensive documentation using Sphinx
- Logging and error handling system
- Extensive test suite
