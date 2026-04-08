# GitHub Push Setup Guide

## Prerequisites
1. Create a GitHub account if you don't have one: https://github.com
2. Install Git: https://git-scm.com
3. Generate SSH key or use Personal Access Token (PAT)

## Option 1: Using SSH (Recommended)

### Generate SSH Key
```bash
ssh-keygen -t ed25519 -C "your-email@example.com"
# Or for older systems:
ssh-keygen -t rsa -b 4096 -C "your-email@example.com"

# Press Enter for default location
# Add passphrase if you want additional security

# Copy public key
cat ~/.ssh/id_ed25519.pub  # Copy the output
# On Windows: type %USERPROFILE%\.ssh\id_ed25519.pub
```

### Add SSH Key to GitHub
1. Go to GitHub → Settings → SSH and GPG keys
2. Click "New SSH key"
3. Paste the copied key
4. Save

### Test SSH Connection
```bash
ssh -T git@github.com
# Should output: Hi username! You've successfully authenticated, but GitHub does not provide shell access.
```

## Option 2: Using Personal Access Token (PAT)

### Generate Token
1. Go to GitHub → Settings → Developer settings → Personal access tokens
2. Click "Generate new token"
3. Select scopes: `repo` (full control of private repositories)
4. Copy the token (you won't be able to see it again)

### Configure Git (Windows)
1. Open Credential Manager
2. Add generic credential:
   - Internet address: `git:https://github.com`
   - Username: `your-github-username`
   - Password: `your-pat-token`

### Configure Git (Mac/Linux)
```bash
# Configure git to cache credentials
git config --global credential.helper store

# Or for 15-minute cache:
git config --global credential.helper cache
```

---

## Push Your Project to GitHub

### Step 1: Create Repository on GitHub

1. Go to https://github.com/new
2. Repository name: `virtual-try-on-ai`
3. Description: `Advanced AI Image Generator for Virtual Try-On`
4. Choose visibility: Public (for portfolio) or Private
5. **Do NOT initialize with README, .gitignore, or license** (already in project)
6. Click "Create repository"

### Step 2: Add Remote and Push

```bash
# Navigate to project
cd "C:\Users\chait\Downloads\Ai image genaretor\virtual-try-on-ai"

# Using SSH (if you set it up)
git remote add origin git@github.com:YOUR-USERNAME/virtual-try-on-ai.git

# OR using HTTPS
git remote add origin https://github.com/YOUR-USERNAME/virtual-try-on-ai.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

### Step 3: Verify

Visit `https://github.com/YOUR-USERNAME/virtual-try-on-ai` to see your uploaded project!

---

## Troubleshooting

### "Permission denied (publickey)"
- Make sure SSH key is added to GitHub
- Check SSH connection: `ssh -T git@github.com`

### "fatal: remote origin already exists"
```bash
git remote remove origin
# Then add again
git remote add origin git@github.com:YOUR-USERNAME/virtual-try-on-ai.git
```

### "fatal: 'origin' does not appear to be a 'git/http' repository"
```bash
# Check current remote
git remote -v

# Verify URL is correct
git remote show origin
```

### Cannot push (authentication fails)
- Using HTTPS? Make sure PAT is in Credential Manager (Windows) or configured (Mac/Linux)
- Using SSH? Verify SSH key is added to GitHub

---

## After Pushing

### Add Badges to README
Edit README.md to add badges:

```markdown
[![GitHub stars](https://img.shields.io/github/stars/YOUR-USERNAME/virtual-try-on-ai?style=social)](https://github.com/YOUR-USERNAME/virtual-try-on-ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
```

### Enable GitHub Pages (Optional)
1. Go to Repository Settings → Pages
2. Source: Deploy from a branch → main
3. Folder: /docs (if you have docs)

### Setup CI/CD (Already Configured!)
GitHub Actions workflow is already in `.github/workflows/ci-cd.yml`
- Automatically runs on push
- Tests on Python 3.8, 3.9, 3.10
- No additional setup needed!

---

## Next Steps

1. **Install Dependencies Locally**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

2. **Prepare Your Dataset**
   - Organize 43,000 images in `datasets/raw/`
   - Follow structure in README.md

3. **Train Model**
   ```bash
   python train.py --config configs/config.yaml --model_type diffusion
   ```

4. **Test API**
   ```bash
   python app.py
   # Visit http://localhost:8000/docs
   ```

5. **Push Updates**
   ```bash
   git add .
   git commit -m "Add description of changes"
   git push
   ```

---

## Useful Git Commands

```bash
# Check status
git status

# View commit history
git log --oneline

# Create feature branch
git checkout -b feature/your-feature-name

# Merge branch
git merge feature-branch-name

# Undo last commit (keep changes)
git reset --soft HEAD~1

# View remote
git remote -v

# Update from remote
git pull origin main
```

---

## Tips for GitHub

✅ **Do**: 
- Commit often with descriptive messages
- Use branches for new features
- Write good docstrings and README
- Add examples and usage instructions
- Keep commits atomic and focused

❌ **Don't**:
- Commit large files (>100MB)
- Commit API keys or passwords
- Commit .venv/ or __pycache__/
- Push directly to main (use branches)

---

**Questions?** Check GitHub docs: https://docs.github.com
