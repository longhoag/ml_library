Contributing
===========

We welcome contributions to ML Library! This document provides guidelines and instructions for contributing.

Development Workflow
------------------

1. Fork the repository and create a feature branch.
2. Make your changes, following the coding style guidelines.
3. Add tests for your changes.
4. Run tests to ensure they pass.
5. Update documentation if necessary.
6. Submit a pull request.

For more details on our git workflow, see our `GIT_WORKFLOW.md` file in the project root.

Coding Style
-----------

We follow these coding guidelines:

* Use Black for code formatting.
* Use isort for import sorting.
* Follow PEP 8 style guidelines.
* Add type hints to all functions and methods.
* Write meaningful docstrings in Google format.

You can use pre-commit hooks to automatically check your code:

.. code-block:: bash

    pre-commit install
    pre-commit run --all-files

Testing
------

All code contributions should include tests:

* Unit tests for new functions or methods.
* Integration tests for more complex features.
* Regression tests for bug fixes.

Run the tests with pytest:

.. code-block:: bash

    pytest

    # For coverage report
    pytest --cov=ml_library tests/

We aim for at least 90% test coverage.

Documentation
------------

Update documentation for any changes:

* Add or update docstrings for all public APIs.
* Update relevant user guides or tutorials.
* For significant changes, add an entry in the changelog.

Build the documentation locally to verify your changes:

.. code-block:: bash

    cd docs
    make html

Pull Request Process
------------------

1. Ensure all tests pass and linting checks succeed.
2. Update documentation and include relevant test cases.
3. Submit the pull request with a clear description of the changes.
4. Address any feedback from code reviews.

License
------

By contributing to ML Library, you agree that your contributions will be licensed under the project's license.
