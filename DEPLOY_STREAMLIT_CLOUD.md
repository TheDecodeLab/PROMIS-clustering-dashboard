# 🚀 Deploy to Streamlit Community Cloud (FREE)

## Quick Deployment Steps

### 1. Go to Streamlit Community Cloud
Visit: **https://share.streamlit.io**

### 2. Sign in with GitHub
- Click "Sign in" 
- Authorize Streamlit to access your GitHub account
- Make sure you're signed in with the account that has access to: `TheDecodeLab/PROMIS-clustering-dashboard`

### 3. Deploy Your App
1. Click **"New app"** button
2. Fill in the deployment form:
   - **Repository**: Select `TheDecodeLab/PROMIS-clustering-dashboard`
   - **Branch**: `master` (or `main` if that's your default)
   - **Main file path**: `app.py`
   - **Python version**: `3.11` (or leave default)
3. Click **"Deploy"**

### 4. Wait for Deployment
- Streamlit will automatically:
  - Install dependencies from `requirements.txt`
  - Run your app
  - Provide you with a public URL

### 5. Share Your App
Once deployed, you'll get a URL like:
```
https://your-app-name.streamlit.app
```

You can share this URL with anyone! 🎉

---

## ✅ Pre-Deployment Checklist

Your repository is already set up correctly:
- ✅ `app.py` exists (main entry point)
- ✅ `requirements.txt` exists with all dependencies
- ✅ `results/dashboard_data.parquet` is included (18MB - perfect size)
- ✅ Repository is on GitHub
- ✅ All code is committed and pushed

---

## 🔄 Updating Your App

Whenever you make changes:
1. Commit your changes: `git add . && git commit -m "Update dashboard"`
2. Push to GitHub: `git push origin master`
3. Streamlit Cloud will **automatically redeploy** your app!

---

## 📝 Notes

- **Free tier**: Unlimited apps for public repositories
- **Auto-deploy**: Every push to your branch triggers a new deployment
- **HTTPS**: Automatically enabled
- **Sleep mode**: Apps sleep after 7 days of inactivity (wake up on first visit)

---

## 🆘 Troubleshooting

If deployment fails:
1. Check the logs in Streamlit Cloud dashboard
2. Verify `requirements.txt` has all dependencies
3. Make sure `app.py` is the correct entry point
4. Check that data file path is relative (not absolute)

---

## 🔗 Alternative Free Options

If Streamlit Cloud doesn't work for you:

1. **Render.com** (Free tier available)
   - Connect GitHub repo
   - Select "Web Service"
   - Build command: `pip install -r requirements.txt`
   - Start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

2. **Railway.app** (Free tier with $5 credit)
   - Connect GitHub repo
   - Auto-detects Python apps
   - Add start command: `streamlit run app.py --server.port=$PORT`

3. **Fly.io** (Free tier available)
   - Requires Dockerfile (you already have one!)
   - More setup but very flexible

---

**Ready to deploy?** Go to https://share.streamlit.io and follow the steps above! 🚀

