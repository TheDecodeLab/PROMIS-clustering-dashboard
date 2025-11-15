# PROMIS Dashboard - Deployment Guide

This guide explains how to prepare and deploy the PROMIS Dashboard for online access with minimal compute and storage requirements.

## 🚀 Quick Start

### 1. Prepare Optimized Data

First, create a lightweight data file from your raw results:

```bash
cd PROMIS_Dashboard
python prepare_dashboard_data.py --source ../results/ --output dashboard_data.parquet
```

This will:
- Process all raw CSV files
- Extract only essential columns needed for visualizations
- Save to `results/dashboard_data.parquet` (typically 10-100x smaller than raw data)
- Enable fast loading in the deployed app

### 2. Test Locally

```bash
streamlit run app.py
```

The app will automatically detect and use the optimized data file if it exists.

## 📦 Deployment Options

### Option 1: Streamlit Cloud (Recommended - Free & Easy)

**Best for:** Quick deployment, free hosting, minimal setup

1. **Create a GitHub repository** with your dashboard:
   ```bash
   cd PROMIS_Dashboard
   git init
   git add .
   git commit -m "Initial dashboard commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set:
     - **Main file path:** `app.py`
     - **Python version:** 3.11 (or your version)
   - Click "Deploy"

3. **Requirements:**
   - Create `requirements.txt` in `PROMIS_Dashboard/`:
     ```
     streamlit>=1.28.0
     pandas>=2.0.0
     numpy>=1.24.0
     plotly>=5.17.0
     scipy>=1.11.0
     pyarrow>=14.0.0
     ```

**Pros:**
- ✅ Free for public repos
- ✅ Automatic HTTPS
- ✅ Easy updates (just push to GitHub)
- ✅ No server management

**Cons:**
- ⚠️ Public repos are public (use private repo for $20/month)
- ⚠️ Limited to 1GB RAM
- ⚠️ Apps sleep after inactivity

---

### Option 2: Heroku (Paid, but Flexible)

**Best for:** More control, private apps, custom domains

1. **Install Heroku CLI** and login

2. **Create `Procfile`:**
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

3. **Create `runtime.txt`:**
   ```
   python-3.11.5
   ```

4. **Deploy:**
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

**Cost:** ~$7/month for basic dyno

---

### Option 3: AWS / Google Cloud / Azure

**Best for:** Enterprise, high traffic, full control

**AWS (EC2 + Elastic Beanstalk):**
- Launch EC2 instance (t2.micro is free tier)
- Install Docker or use Elastic Beanstalk
- Deploy with Dockerfile

**Google Cloud Run:**
- Containerize app with Docker
- Deploy to Cloud Run (pay per use)
- Very cost-effective for low traffic

**Azure App Service:**
- Similar to Heroku but Microsoft ecosystem
- Good integration with other Azure services

---

### Option 4: Docker + Any Cloud Provider

**Best for:** Maximum portability

1. **Create `Dockerfile`:**
   ```dockerfile
   FROM python:3.11-slim
   
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   COPY . .
   
   EXPOSE 8501
   
   HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
   
   ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Build and run:**
   ```bash
   docker build -t promis-dashboard .
   docker run -p 8501:8501 promis-dashboard
   ```

3. **Deploy to:**
   - AWS ECS/Fargate
   - Google Cloud Run
   - Azure Container Instances
   - DigitalOcean App Platform
   - Railway.app
   - Render.com

---

## 📊 Data File Optimization

The optimized `dashboard_data.parquet` file contains only:

- **Metrics:** silhouette_score, davies_bouldin_index, calinski_harabasz_index, wss
- **Metadata:** method, dr_method, variant, panel, stratification
- **Parameters:** n_clusters, dr_components
- **Optional:** filename, dataname (for feature importance)

**Typical size reduction:**
- Raw CSV files: 500MB - 5GB
- Optimized Parquet: 10MB - 100MB (50-100x smaller!)

## 🔒 Security Considerations

1. **Data Privacy:** Ensure your data file doesn't contain sensitive patient information
2. **Access Control:** 
   - Use private GitHub repos for Streamlit Cloud
   - Add authentication for production deployments
3. **Environment Variables:** Store sensitive paths/configs in env vars

## 🛠️ Troubleshooting

### App won't load data
- Check that `results/dashboard_data.parquet` exists
- Verify file permissions
- Check file size (should be > 0 bytes)

### Deployment fails
- Ensure `requirements.txt` includes all dependencies
- Check Python version compatibility
- Verify file paths are relative (not absolute)

### Slow loading
- Use optimized Parquet file instead of raw CSVs
- Reduce data size by filtering before preparation
- Consider pagination for very large datasets

## 📝 Example Deployment Script

```bash
#!/bin/bash
# deploy.sh - Prepare and deploy dashboard

echo "Preparing optimized data..."
python prepare_dashboard_data.py --source ../results/

echo "Testing locally..."
streamlit run app.py --server.headless=true &
sleep 5
curl http://localhost:8501/_stcore/health
kill %1

echo "Ready to deploy!"
echo "1. Commit changes: git add . && git commit -m 'Update dashboard'"
echo "2. Push to GitHub: git push"
echo "3. Streamlit Cloud will auto-deploy"
```

---

## 💡 Tips

1. **Version Control:** Keep `dashboard_data.parquet` in `.gitignore` if it's large (>100MB)
2. **Updates:** Re-run `prepare_dashboard_data.py` when you have new results
3. **Monitoring:** Use Streamlit Cloud's built-in analytics or add custom logging
4. **Backup:** Keep a copy of the optimized data file in cloud storage

---

## 🆘 Need Help?

- Streamlit docs: https://docs.streamlit.io
- Streamlit Cloud: https://docs.streamlit.io/streamlit-community-cloud
- GitHub Issues: Create an issue in your repo

