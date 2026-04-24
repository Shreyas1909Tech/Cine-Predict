# 🚀 CinePredict - Complete Render Deployment Guide

## ✅ Automated Deployment Setup (5 minutes)

All necessary files have been added to the repository for automatic Render deployment.

---

## 📋 Files Added for Render Deployment

```
✅ Procfile                    - Render web process configuration
✅ runtime.txt                 - Python version specification
✅ render.yaml                 - Render service configuration
✅ build.sh                    - Build script for dependencies
✅ .github/workflows/deploy-render.yml - Auto-deploy on GitHub push
✅ backend/api.py (updated)    - Render-optimized FastAPI app
```

---

## 🎯 Step-by-Step Render Deployment

### STEP 1: Go to Render Dashboard
```
1. Open https://render.com
2. Sign up (free account)
3. Connect your GitHub account
```

### STEP 2: Create Backend Service
```
1. Click "New +" button
2. Select "Web Service"
3. Connect GitHub repo: Shreyas1909Tech/Cine-Predict
4. Configure as follows:
```

**Backend Configuration:**

| Setting | Value |
|---------|-------|
| **Name** | cinepredict-backend |
| **Environment** | Python 3 |
| **Region** | Choose closest to you |
| **Branch** | main |
| **Build Command** | `bash build.sh` |
| **Start Command** | `uvicorn backend.api:app --host 0.0.0.0 --port $PORT` |
| **Plan** | Free |

### STEP 3: Set Environment Variables (Optional)
```
No critical environment variables needed for MVP
But you can add:

PYTHONUNBUFFERED=true
RENDER=true
```

### STEP 4: Deploy
```
1. Click "Create Web Service"
2. Wait for deployment (2-3 minutes)
3. Check logs for any errors
4. Get your API URL: https://cinepredict-backend.onrender.com
```

### STEP 5: Create Frontend Service
```
1. Click "New +" → "Static Site"
2. Connect same GitHub repo
3. Configure as follows:
```

**Frontend Configuration:**

| Setting | Value |
|---------|-------|
| **Name** | cinepredict-frontend |
| **Region** | Same as backend |
| **Branch** | main |
| **Build Command** | (leave empty - no build needed) |
| **Publish Directory** | `frontend` |

### STEP 6: Connect Frontend to Backend
```
After frontend is deployed:

1. Get frontend URL: https://cinepredict-frontend.onrender.com
2. Get backend URL: https://cinepredict-backend.onrender.com
3. Update frontend/index.html:
   - Find: const API_BASE = 'http://localhost:8000'
   - Replace with: const API_BASE = 'https://cinepredict-backend.onrender.com'
4. Commit and push to GitHub
```

### STEP 7: Test Deployment
```
1. Open frontend: https://cinepredict-frontend.onrender.com
2. Test prediction feature
3. Check API docs: https://cinepredict-backend.onrender.com/docs
4. Verify health check: https://cinepredict-backend.onrender.com/health
```

---

## 🔄 Automatic Deployment (GitHub Actions)

When you push to GitHub, both services auto-redeploy:

```bash
git add .
git commit -m "Update deployment configuration"
git push origin main
```

**Render automatically:**
1. Pulls latest code
2. Runs build.sh
3. Installs dependencies
4. Downloads NLTK data
5. Checks/trains models
6. Starts API server

---

## 📊 Deployment Architecture

```
┌─────────────────────────────────────────────────────┐
│                   GitHub Repository                  │
│            (Shreyas1909Tech/Cine-Predict)           │
└──────────────────┬──────────────────────────────────┘
                   │
         ┌─────────┴─────────┐
         ↓                   ↓
   ┌──────────────┐    ┌──────────────┐
   │   Render     │    │   Render     │
   │   Frontend   │    │   Backend    │
   │ (Static Site)│    │ (Web Service)│
   └──────────────┘    └──────────────┘
         ↓                   ↓
   HTTPS Frontend      HTTPS API
   (CDN Cached)        (Auto-scaled)
```

---

## 🔍 Monitoring Your Deployment

### Check Backend Status
```
API Health: https://cinepredict-backend.onrender.com/health
API Docs: https://cinepredict-backend.onrender.com/docs
Render Dashboard: https://dashboard.render.com
```

### View Logs
```
1. Go to Render Dashboard
2. Click on "cinepredict-backend"
3. View real-time logs
4. Check for errors
```

### Common Issues & Solutions

**Issue: Build fails with "Models not found"**
```
Solution: Models will be auto-generated on first deployment
This adds 5-10 minutes to build time (only first time)
Subsequent deployments are faster
```

**Issue: Frontend API calls return 404**
```
Solution: Update API_BASE URL in frontend/index.html
Must match backend Render URL exactly
```

**Issue: CORS errors**
```
Solution: Backend already configured for CORS
Allows requests from any frontend origin
```

**Issue: Service spins down (free tier)**
```
Note: Free tier spins down inactive services after 15 minutes
First request after spin-down takes 30 seconds to respond
Upgrade to paid plan for always-on service
```

---

## 📈 Performance & Scaling

### Current (Free Tier)
```
Backend: Shared CPU, 512MB RAM
Frontend: CDN cached, instant load
Auto-downscale after 15 minutes inactivity
Cold start time: ~30 seconds
```

### After Upgrade (Paid)
```
Backend: Dedicated CPU, 1GB+ RAM, always on
Frontend: Global CDN
Auto-scale on load
Response time: <100ms
```

---

## 💾 Model Management

Models are stored in repository and deployed with code:

```
backend/models/
├── regression_model.pkl          (~50MB)
├── classification_model.pkl      (~30MB)
├── feature_columns.pkl           (~1KB)
├── label_encoder.pkl             (~1KB)
├── metrics.json                  (~1KB)
└── feature_importance.json       (~2KB)
```

**Total size: ~85MB** (Well within GitHub limits)

To update models:
```bash
# Train locally
python backend/train_model.py

# Commit new models
git add backend/models/
git commit -m "Update trained models"
git push origin main

# Render auto-deploys
```

---

## 🔐 Security Considerations

### Current Setup (Suitable for MVP)
- ✅ HTTPS enabled (Render certificate)
- ✅ CORS configured for public access
- ✅ No authentication required
- ⚠️ Anyone can use the API

### For Production
Add these security measures:

```python
# In backend/api.py
from fastapi.security import APIKeyHeader

security = APIKeyHeader(name="X-API-Key")

@app.post("/predict")
async def predict(req: PredictRequest, api_key: str = Depends(security)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")
    # ... rest of prediction logic
```

Then set environment variable in Render:
```
API_KEY=your-secret-key-here
```

---

## 📱 Testing Deployed Application

### Quick Test
```bash
# Test backend health
curl https://cinepredict-backend.onrender.com/health

# Test prediction API
curl -X POST https://cinepredict-backend.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Test Movie",
    "budget": 100000000,
    "runtime": 120,
    "release_month": 7,
    "release_year": 2025,
    "genres": ["Action"],
    "cast_popularity": 70,
    "director_popularity": 60,
    "popularity": 65,
    "vote_average": 7.5,
    "plot_overview": "An amazing action adventure"
  }'
```

### Frontend Testing
```
1. Open https://cinepredict-frontend.onrender.com
2. Fill movie details
3. Click "PREDICT BOX OFFICE"
4. Verify results display
5. Check all pages work
```

---

## 🎬 Final Deployment Checklist

- [ ] Repository pushed to GitHub (Shreyas1909Tech/Cine-Predict)
- [ ] Procfile created
- [ ] runtime.txt created
- [ ] render.yaml created
- [ ] build.sh created
- [ ] GitHub Actions workflow created
- [ ] Backend deployed on Render
- [ ] Frontend deployed on Render
- [ ] Frontend API_BASE URL updated
- [ ] Health endpoint tested
- [ ] Prediction API tested
- [ ] Frontend page loads
- [ ] All pages accessible
- [ ] Charts render correctly
- [ ] Sentiment analysis works
- [ ] No console errors

---

## 📊 Deployment Summary

```
╔════════════════════════════════════════════════════════════════╗
║                   DEPLOYMENT STATUS                           ║
╠════════════════════════════════════════════════════════════════╣
║ Backend API        https://cinepredict-backend.onrender.com   ║
║ Frontend App       https://cinepredict-frontend.onrender.com  ║
║ API Documentation  /docs endpoint                             ║
║ Health Check       /health endpoint                           ║
║ Total Setup Time   5-10 minutes                               ║
║ Cost               FREE (with limitations)                    ║
║ Auto-Deploy        Yes (on GitHub push)                       ║
║ Scaling            Manual (free → paid)                       ║
║ Always-On          No (spins down after 15min)                ║
╚════════════════════════════════════════════════════════════════╝
```

---

## 🚀 Next Steps

### Immediate
1. ✅ Deploy using this guide
2. ✅ Test all functionality
3. ✅ Share URLs with team

### Short Term (Week 1)
1. Add API authentication
2. Setup error monitoring (Sentry)
3. Add database (PostgreSQL)
4. Implement logging

### Medium Term (Month 1)
1. Upgrade to paid Render plan
2. Setup CI/CD pipeline improvements
3. Add caching layer (Redis)
4. Performance optimization

### Long Term
1. Multi-region deployment
2. Load balancing
3. Advanced monitoring
4. Cost optimization

---

## 📞 Support & Troubleshooting

### Common Commands

```bash
# Check if files are committed
git status

# View Render logs
# Go to: https://dashboard.render.com → Service → Logs

# Manually redeploy
# Go to: Render Dashboard → Service → Manual Deploy

# Clear build cache
# Go to: Render Dashboard → Settings → Clear Build Cache
```

### Resources
- Render Documentation: https://render.com/docs
- FastAPI Documentation: https://fastapi.tiangolo.com
- CinePredict GitHub: https://github.com/Shreyas1909Tech/Cine-Predict

---

## 🎉 Congratulations!

Your CinePredict application is now live on Render! 

**Share your deployment:**
```
🎬 CinePredict is now live!
Frontend: https://cinepredict-frontend.onrender.com
API: https://cinepredict-backend.onrender.com

Predict movie box office success using AI! 🤖
```

---

**Last Updated:** 2024
**Status:** ✅ Ready for Production MVP
