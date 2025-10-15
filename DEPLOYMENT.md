# 🚀 Sudoku Solver Deployment Guide

This guide will help you deploy your Sudoku Solver app using Vercel (frontend) + Railway (backend).

## 📋 Prerequisites

- GitHub account
- Vercel account (free)
- Railway account (free)

## 🎯 Deployment Steps

### Step 1: Deploy Backend to Railway

1. **Push your code to GitHub:**
   ```bash
   git add .
   git commit -m "Add deployment configuration"
   git push origin main
   ```

2. **Deploy to Railway:**
   - Go to [railway.app](https://railway.app)
   - Sign in with GitHub
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your Sudoku repository
   - Railway will automatically detect the Python backend

3. **Configure Railway:**
   - Set the root directory to `backend/`
   - Railway will install dependencies from `backend/requirements.txt`
   - The app will start with `python app.py`

4. **Get your backend URL:**
   - Railway will provide a URL like `https://your-app-name.railway.app`
   - Copy this URL for the next step

### Step 2: Deploy Frontend to Vercel

1. **Deploy to Vercel:**
   - Go to [vercel.com](https://vercel.com)
   - Sign in with GitHub
   - Click "New Project" → "Import Git Repository"
   - Select your Sudoku repository

2. **Configure Vercel:**
   - **Framework Preset:** Create React App
   - **Root Directory:** `frontend`
   - **Build Command:** `npm run build`
   - **Output Directory:** `build`

3. **Set Environment Variables:**
   - In Vercel dashboard, go to Settings → Environment Variables
   - Add: `REACT_APP_API_URL` = `https://your-backend-url.railway.app/api`
   - Make sure to include `/api` at the end

4. **Deploy:**
   - Click "Deploy"
   - Vercel will build and deploy your React app

### Step 3: Test Your Deployment

1. **Frontend URL:** `https://your-app-name.vercel.app`
2. **Backend URL:** `https://your-backend-url.railway.app/api/health`

## 🔧 Alternative Backend Deployment Options

### Option A: Render.com
1. Connect GitHub repository
2. Choose "Web Service"
3. Set:
   - **Build Command:** `pip install -r backend/requirements.txt`
   - **Start Command:** `cd backend && python app.py`
   - **Environment:** Python 3

### Option B: Heroku
1. Install Heroku CLI
2. Create `Procfile` (already created)
3. Deploy:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

## 🎨 Custom Domain (Optional)

### Vercel Custom Domain:
1. Go to Vercel dashboard → Settings → Domains
2. Add your custom domain
3. Update DNS records as instructed

### Railway Custom Domain:
1. Go to Railway dashboard → Settings → Domains
2. Add your custom domain
3. Update DNS records

## 🔍 Troubleshooting

### Common Issues:

1. **CORS Errors:**
   - Make sure Flask-CORS is installed
   - Check that `CORS(app)` is in your backend code

2. **API Not Found:**
   - Verify the API URL includes `/api` at the end
   - Check that environment variables are set correctly

3. **Build Failures:**
   - Check that all dependencies are in `requirements.txt`
   - Verify Python version compatibility

4. **File Upload Issues:**
   - Ensure file size limits are appropriate
   - Check that multipart form data is handled correctly

## 📊 Monitoring

### Vercel Analytics:
- Built-in analytics for frontend performance
- View in Vercel dashboard

### Railway Monitoring:
- Built-in logs and metrics
- View in Railway dashboard

## 🔄 Updates

To update your deployment:
1. Push changes to GitHub
2. Vercel and Railway will automatically redeploy
3. No manual intervention needed

## 💰 Cost

- **Vercel:** Free tier includes 100GB bandwidth/month
- **Railway:** Free tier includes $5 credit/month
- **Total:** Free for small to medium usage

## 🎯 Production Checklist

- [ ] Environment variables set correctly
- [ ] CORS configured properly
- [ ] Error handling implemented
- [ ] Logging configured
- [ ] Health check endpoint working
- [ ] Custom domain configured (optional)
- [ ] SSL certificates active
- [ ] Performance monitoring enabled

Your Sudoku Solver app is now live! 🎉
