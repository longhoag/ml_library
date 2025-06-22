# PyPI Token Instructions

This document provides instructions on how to generate and use PyPI API tokens for package publishing.

## Generating a PyPI Token

1. Create an account on [PyPI](https://pypi.org/) if you don't already have one.

2. Go to your account settings by clicking on your username in the top-right corner and selecting "Account settings".

3. Scroll down to the "API tokens" section and click on "Add API token".

4. Give your token a name (e.g., "ml_library-publish") and select the scope:
   - For publishing only to this project, select "Scope: Project" and choose "ml_library".
   - If you want to publish multiple projects with the same token, you can select "Scope: All projects".

5. Click "Create token" and save the token value somewhere secure.
   **Important**: You won't be able to see this token again after you leave the page.

## Using the Token for Manual Publishing

When publishing manually with Twine:

```bash
python -m twine upload --username __token__ --password <your-token> dist/*
```

## Setting Up the Token in GitHub Actions

To use the token with GitHub Actions:

1. Go to your GitHub repository settings.

2. Click on "Secrets and variables" -> "Actions" in the left sidebar.

3. Click "New repository secret".

4. Name: `PYPI_API_TOKEN`
   Value: Your PyPI token (paste the token value)

5. Click "Add secret".

Now the GitHub Actions workflow will be able to use this secret for automated publishing.

## Test PyPI Token

If you want to test your publishing process before releasing to the main PyPI, you can:

1. Create an account on [Test PyPI](https://test.pypi.org/)
2. Generate a token there as well
3. Add it as a separate GitHub secret named `TEST_PYPI_API_TOKEN`

Then modify the GitHub workflow or publishing script to use Test PyPI first.
