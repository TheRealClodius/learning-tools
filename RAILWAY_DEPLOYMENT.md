# Railway Deployment Guide - Signal AI Agent

## 🚀 Quick Deploy

### Method 1: Full Feature Deploy (Recommended)
```bash
# 1. Connect to Railway
railway login
railway init

# 2. Set environment variables in Railway dashboard
ANTHROPIC_API_KEY=your_key_here
SLACK_BOT_TOKEN=xoxb-your-token
SLACK_SIGNING_SECRET=your_secret

# 3. Deploy
railway up
```

### Method 2: Simplified Deploy (If Method 1 fails)
If you encounter build issues with heavy ML dependencies:

```bash
# Use the simplified requirements file
cp requirements-simple.txt requirements.txt
railway up
```

## 🔧 Configuration Files

The following files have been optimized for Railway:

- `nixpacks.toml` - Build configuration with system dependencies
- `railway.toml` - Railway-specific deployment settings  
- `Procfile` - Process configuration
- `slack_webhook_server.py` - Updated with proper port binding

## 🚨 Common Issues & Solutions

### Issue 1: Build Timeout or Memory Issues
**Symptoms**: Build fails with timeout or out-of-memory errors
**Cause**: `faiss-cpu` and `sentence-transformers` require compilation
**Solutions**:
1. Use the simplified requirements: `cp requirements-simple.txt requirements.txt`
2. Increase build resources in Railway dashboard
3. Use pre-compiled Docker image (advanced)

### Issue 2: Port Binding Issues
**Symptoms**: App deployed but not accessible, "PORT not found" errors
**Solution**: Updated `slack_webhook_server.py` now includes proper port binding with `PORT` environment variable

### Issue 3: Python Version Conflicts
**Symptoms**: Build fails with distutils or packaging errors
**Solution**: Standardized on Python 3.11 across all config files

### Issue 4: Dependency Resolution Conflicts
**Symptoms**: pip resolver conflicts during install
**Solutions**:
1. Updated `requirements.txt` with missing dependencies
2. Added pip timeout and cache settings in `nixpacks.toml`

## 📝 Environment Variables Required

Set these in your Railway dashboard:

```
ANTHROPIC_API_KEY=sk-ant-...
SLACK_BOT_TOKEN=xoxb-...
SLACK_SIGNING_SECRET=...
PORT=8000  # Usually set automatically by Railway
```

## 🔍 Health Check

The app now includes a health check endpoint at `/` that returns:
```json
{"status": "healthy", "service": "signal-ai-agent"}
```

## 📋 Deployment Checklist

- [ ] Railway CLI installed and logged in
- [ ] Environment variables set in Railway dashboard
- [ ] Repository connected to Railway project
- [ ] Build completes successfully (check logs)
- [ ] Health check endpoint responds
- [ ] Slack webhook endpoints accessible

## 🆘 If All Else Fails

1. **Check Railway logs**: `railway logs`
2. **Try local build**: `docker build -t test .` (if you have Docker)
3. **Use minimal deployment**: Deploy with `requirements-simple.txt`
4. **Contact support**: Include Railway build logs and error messages

## 🔄 Rollback Strategy

If deployment fails:
```bash
# Revert to previous working version
git revert HEAD
railway up

# Or use simplified requirements
cp requirements-simple.txt requirements.txt
railway up
```