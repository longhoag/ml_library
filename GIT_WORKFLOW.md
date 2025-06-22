# Git Workflow

This document outlines the Git workflow to be followed for this project.

## Branching Strategy

We follow a modified Git Flow approach:

- `main`: Production-ready code
- `develop`: Latest development changes
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `release/*`: Release preparation
- `hotfix/*`: Urgent production fixes

## Workflow Steps

### Starting a New Feature

1. Create a feature branch from `develop`:
```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

2. Make your changes, commit often:
```bash
git add .
git commit -m "Descriptive message"
```

3. Push your branch to remote:
```bash
git push -u origin feature/your-feature-name
```

### Pull Request Process

1. Open a Pull Request (PR) to merge your feature branch into `develop`
2. Ensure the PR description details the changes and any relevant information
3. Link any related issues to the PR
4. Request code reviews from at least one team member
5. Ensure all CI checks pass

### Code Review Guidelines

- Review code for clarity, correctness, and adherence to project standards
- Provide constructive feedback
- Verify test coverage is adequate
- Check documentation is updated

### Merging

Once approved and all checks pass:

1. Squash and merge the feature branch into `develop`
2. Delete the feature branch after successful merge

### Releases

1. Create a release branch from `develop`:
```bash
git checkout develop
git checkout -b release/vX.Y.Z
```

2. Make any final release adjustments and version bumps
3. Create a PR to merge into `main` and back to `develop`
4. After merging to `main`, tag the release:
```bash
git checkout main
git pull origin main
git tag -a vX.Y.Z -m "Version X.Y.Z"
git push origin vX.Y.Z
```

## CI Hooks

Pre-commit hooks will run:
- Code linting
- Static analysis
- Unit tests

CI pipeline will run on all PRs with:
- Full test suite
- Coverage reporting
- Documentation building
