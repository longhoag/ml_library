Installation
============

You can install ML Library using pip:

.. code-block:: bash

    pip install ml-library

Development Installation
-----------------------

To install the development version with all dependencies:

1. Clone the repository:

   .. code-block:: bash

       git clone https://github.com/yourusername/ml_library.git
       cd ml_library

2. Create and activate a virtual environment:

   .. code-block:: bash

       python -m venv venv
       source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install development dependencies:

   .. code-block:: bash

       pip install -e ".[dev]"

4. (Optional) Set up pre-commit hooks:

   .. code-block:: bash

       pre-commit install

Dependencies
-----------

ML Library requires:

* Python >= 3.8
* numpy
* pandas
* scikit-learn
* matplotlib

For development, additional dependencies are required:

* pytest
* pytest-cov
* flake8
* black
* isort
* mypy
* sphinx
* sphinx-rtd-theme
* pre-commit
