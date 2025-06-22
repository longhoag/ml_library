#!/bin/bash

# Script to update GitHub token with workflow permissions
echo "This script will help you update your GitHub credentials for this repository."
echo "Follow these steps to create a new Personal Access Token (PAT) with workflow permissions:"
echo ""
echo "1. Go to GitHub: https://github.com/settings/tokens"
echo "2. Click 'Generate new token' > 'Generate new token (classic)'"
echo "3. Name it 'ML Library Workflow Token' or similar"
echo "4. Set expiration as needed"
echo "5. Select these scopes: repo, workflow, read:packages, write:packages"
echo "6. Click 'Generate token'"
echo "7. Copy the generated token"
echo ""
echo "After you've created the token, paste it below when prompted."
echo ""

read -p "Enter your GitHub username: " USERNAME
read -sp "Paste your new GitHub token: " TOKEN
echo ""

# Update the remote URL with the new token
git remote set-url origin https://$USERNAME:$TOKEN@github.com/longhoag/ml_library.git

echo ""
echo "Remote URL updated with the new token!"
echo "Test it by pushing a change to GitHub."
echo "You can now try: git push origin testing"
echo ""
echo "Note: This token is now stored in your Git config. Keep this repository secure."
