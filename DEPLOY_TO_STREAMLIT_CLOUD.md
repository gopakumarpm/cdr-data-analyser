# How to Deploy CDR Analyser to Streamlit Cloud (FREE)

## Step 1: Create a GitHub Account (if you don't have one)
1. Go to https://github.com
2. Click "Sign Up" and create a free account

## Step 2: Create a New Repository
1. Click the "+" icon in top right → "New repository"
2. Name it: `cdr-analyzer`
3. Make it **Public** (required for free Streamlit Cloud)
4. Click "Create repository"

## Step 3: Upload Files to GitHub
Upload these files from `D:\Data-science\Claude\CDR Data Analyser\streamlit_cloud\`:

```
cdr-analyzer/
├── app.py
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml
```

**Method 1: Using GitHub Web Interface**
1. In your new repo, click "Add file" → "Upload files"
2. Drag and drop all files from the `streamlit_cloud` folder
3. Click "Commit changes"

**Method 2: Using Git Command Line**
```bash
cd "D:\Data-science\Claude\CDR Data Analyser\streamlit_cloud"
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/cdr-analyzer.git
git push -u origin main
```

## Step 4: Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io
2. Click "Sign in with GitHub"
3. Click "New app"
4. Select:
   - Repository: `YOUR_USERNAME/cdr-analyzer`
   - Branch: `main`
   - Main file path: `app.py`
5. Click "Deploy!"

## Step 5: Wait for Deployment
- Deployment takes 2-5 minutes
- Once done, you'll get a URL like: `https://your-app-name.streamlit.app`

## Step 6: Share the URL
- Share this URL with anyone
- They can access the analyzer without any installation!

---

## Your App URL Will Be:
`https://cdr-analyzer-YOUR_USERNAME.streamlit.app`

## Benefits of Streamlit Cloud:
✅ **Free** - No cost for public apps
✅ **No Installation** - Users just open the URL
✅ **Always Available** - 24/7 access
✅ **Auto Updates** - Push to GitHub, app updates automatically
✅ **Secure** - HTTPS by default

## Limitations (Free Tier):
- App sleeps after 7 days of inactivity (wakes up when accessed)
- 1GB RAM limit
- Public repository required

---

## Quick Git Commands (if needed):

### First time setup:
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### To update the app later:
```bash
cd "D:\Data-science\Claude\CDR Data Analyser\streamlit_cloud"
git add .
git commit -m "Update app"
git push
```

The app will automatically redeploy when you push changes!
