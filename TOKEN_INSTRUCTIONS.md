# GitHub Token Update Instructions

Follow these steps to update your GitHub token with workflow permissions:

## Step 1: Create a new Personal Access Token (PAT)

1. Go to GitHub: https://github.com/settings/tokens
2. Click "Generate new token" > "Generate new token (classic)"
3. For "Note", enter "ML Library Workflow Token"
4. Set an appropriate expiration date
5. Select these permission scopes:
   - ✅ `repo` (all repo permissions)
   - ✅ `workflow`
   - ✅ `read:packages` (optional, if you plan to publish packages)
   - ✅ `write:packages` (optional, if you plan to publish packages)
6. Click "Generate token"
7. **IMPORTANT**: Copy the generated token immediately!

## Step 2: Update your local repository

You have two options:

### Option 1: Use the helper script

Run the provided helper script:
```bash
./update_github_token.sh
```

The script will prompt you for your GitHub username and new token.

### Option 2: Update manually

Set your remote URL with the new token:
```bash
git remote set-url origin https://USERNAME:NEW_TOKEN@github.com/longhoag/ml_library.git
```
Replace `USERNAME` with your GitHub username and `NEW_TOKEN` with your new personal access token.

## Step 3: Test the token

After updating your token, test it by pushing to GitHub:
```bash
git add .github/workflows
git commit -m "Add GitHub Actions workflow files"
git push origin testing
```

If the push succeeds, your token now has workflow permissions!

## Security Note

The token is stored in your Git configuration. Make sure to keep your development environment secure, as anyone with access to your computer could potentially use this token.

If you're using a shared or public computer, consider removing the token when you're done:
```bash
git remote set-url origin https://github.com/longhoag/ml_library.git
```
