# 🚀 Quick Deployment Guide

## Step-by-Step Deployment Instructions

### Prerequisites Checklist
- [ ] All model files are in the project root
- [ ] Python 3.11+ installed
- [ ] Git repository initialized
- [ ] GitHub/GitLab account (for Netlify)

---

## Option 1: Railway (Backend) + Netlify (Frontend) - RECOMMENDED

### Part A: Deploy Backend to Railway (20 minutes)

1. **Sign up for Railway**
   - Go to https://railway.app
   - Sign up with GitHub

2. **Create New Project**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

3. **Configure Railway**
   - Railway auto-detects Python
   - Add environment variable (if needed): `PORT=5000`
   - Railway will run: `gunicorn app:app`

4. **Get Your Backend URL**
   - After deployment, Railway provides a URL like: `https://your-app.up.railway.app`
   - Copy this URL

5. **Update Frontend API URL**
   - Edit `static/js/script.js`
   - Change line 2: `const API_BASE_URL = 'https://your-app.up.railway.app';`
   - Save and commit

### Part B: Deploy Frontend to Netlify (5 minutes)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Deploy to Netlify**
   - Go to https://app.netlify.com
   - Click "Add new site" → "Import an existing project"
   - Connect to GitHub and select your repo

3. **Configure Netlify**
   - Build command: `echo "No build step"`
   - Publish directory: `static`
   - Click "Deploy site"

4. **Done!** Your site will be live at `https://your-site.netlify.app`

---

## Option 2: Render (Full Stack) - ALTERNATIVE

### Deploy Everything to Render (30 minutes)

1. **Sign up for Render**
   - Go to https://render.com
   - Sign up with GitHub

2. **Create Web Service**
   - Click "New" → "Web Service"
   - Connect your GitHub repository

3. **Configure Settings**
   - **Name:** `ai-translator`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Instance Type:** Free tier (or paid for better performance)

4. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment (10-20 minutes)
   - Your app will be live!

---

## Option 3: Local Testing First

### Test Locally Before Deploying (10 minutes)

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the App**
   ```bash
   python app.py
   ```

3. **Test in Browser**
   - Open http://localhost:5000
   - Try translating some text
   - Verify everything works

4. **If it works locally, proceed with deployment!**

---

## Troubleshooting

### Models Not Loading?
- Check all `.keras` and `.json` files are in root directory
- Verify file paths in `app.py` match your structure

### CORS Errors?
- Backend already has CORS enabled
- If issues persist, check `flask-cors` is installed

### Deployment Fails?
- Check `requirements.txt` has all dependencies
- Verify Python version (3.11+)
- Check Railway/Render logs for errors

### Frontend Can't Connect to Backend?
- Update `API_BASE_URL` in `static/js/script.js`
- Ensure backend URL is correct
- Check CORS is enabled on backend

---

## Quick Commands Reference

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py

# Test API endpoint
curl -X POST http://localhost:5000/api/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello","language":"French"}'

# Check health
curl http://localhost:5000/api/health
```

---

## Post-Deployment Checklist

- [ ] Backend is accessible (check `/api/health`)
- [ ] Frontend loads correctly
- [ ] Translation works for French
- [ ] Translation works for Spanish
- [ ] Error handling works
- [ ] Mobile responsive design works

---

## Need Help?

If you encounter issues:
1. Check the logs in Railway/Render dashboard
2. Verify all model files are uploaded
3. Test API endpoints directly
4. Check browser console for frontend errors

Good luck with your deployment! 🚀
