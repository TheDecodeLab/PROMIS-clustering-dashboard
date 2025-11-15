# 🚀 Quick Start Guide

## Step 1: Prepare Optimized Data

Create a lightweight data file from your raw results:

```bash
cd PROMIS_Dashboard
python prepare_dashboard_data.py --source ../results/
```

This creates `results/dashboard_data.parquet` (typically 50-100x smaller than raw data).

## Step 2: Test Locally

```bash
streamlit run app.py
```

The app will automatically use the optimized data file if it exists.

## Step 3: Deploy Online

### Easiest Option: Streamlit Cloud (Free)

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "PROMIS Dashboard"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repo
   - Main file: `app.py`
   - Click "Deploy"

3. **Done!** Your dashboard will be live in ~2 minutes.

### Alternative: Docker Deployment

```bash
# Build image
docker build -t promis-dashboard .

# Run locally
docker run -p 8501:8501 promis-dashboard

# Or deploy to any cloud provider that supports Docker
```

## 📊 What Gets Deployed?

- ✅ `app.py` - Main application
- ✅ `interactive_metrics_viz.py` - Standalone visualization code
- ✅ `results/dashboard_data.parquet` - Optimized data (if < 100MB, include in git)
- ✅ `requirements.txt` - Dependencies

## 💡 Tips

- **Data size:** If `dashboard_data.parquet` > 100MB, add it to `.gitignore` and upload separately
- **Updates:** Re-run `prepare_dashboard_data.py` when you have new results
- **Privacy:** Ensure your data doesn't contain sensitive information

## 🆘 Troubleshooting

**Import errors?** Make sure all files are in the same directory.

**Data not loading?** Check that `results/dashboard_data.parquet` exists and is readable.

**Deployment fails?** Check `requirements.txt` has all dependencies.

For more details, see [README_DEPLOYMENT.md](README_DEPLOYMENT.md)

