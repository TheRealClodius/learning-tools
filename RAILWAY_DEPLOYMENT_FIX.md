# Railway Deployment Fix Guide

## Problem Summary
The Railway deployment was failing with the error:
```
/bin/bash: line 1: pip: command not found
```

This occurred because Nixpacks wasn't properly setting up the Python environment with pip available in the PATH.

## Solution Implemented

### 1. **Switched from Nixpacks to Docker**
Created a `Dockerfile` that uses the official Python 3.11 slim image, which includes pip by default:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*
COPY requirements-minimal.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-minimal.txt
COPY . .
EXPOSE 8080
CMD ["python", "slack_webhook_server.py"]
```

### 2. **Updated railway.toml**
Changed the builder from nixpacks to dockerfile:

```toml
[build]
builder = "dockerfile"
dockerfilePath = "Dockerfile"
```

### 3. **Created Minimal Requirements**
Created `requirements-minimal.txt` without heavy ML dependencies for testing:
- Removed: faiss-cpu, sentence-transformers, numpy, torch, transformers, memoryos-pro
- Kept: Core web framework and API dependencies

## Deployment Steps

1. **Push to GitHub**: The changes have been pushed to the branch `cursor/fix-pip-command-not-found-during-deployment-a4f2`

2. **Railway Deployment**: Railway should automatically detect the new push and start a new deployment using Docker

3. **Monitor Deployment**: Check the Railway dashboard for build progress

## Next Steps After Successful Deployment

Once the minimal deployment works:

1. **Gradually Add Dependencies**: 
   - Test adding back ML dependencies one by one
   - Consider using a multi-stage Docker build for optimization

2. **Alternative Approach**: If you need the ML dependencies:
   ```dockerfile
   # Use a larger base image with more build tools
   FROM python:3.11
   
   # Install additional system dependencies for ML libraries
   RUN apt-get update && apt-get install -y \
       gcc g++ gfortran cmake pkg-config \
       liblapack-dev libopenblas-dev \
       && rm -rf /var/lib/apt/lists/*
   ```

3. **Environment Variables**: Ensure all required environment variables are set in Railway:
   - `PORT` (Railway sets this automatically)
   - Any API keys or configuration values

## Troubleshooting

If deployment still fails:

1. **Check Railway Logs**: Look for specific error messages
2. **Test Locally**: 
   ```bash
   docker build -t test-app .
   docker run -p 8080:8080 test-app
   ```
3. **Verify Dependencies**: Ensure all packages in requirements exist and are compatible

## Alternative Nixpacks Configuration (if needed)

If you prefer to stick with Nixpacks, the updated `nixpacks.toml` creates a virtual environment:

```toml
[phases.setup]
nixPkgs = ["python311", "gcc"]

[phases.install]
cmds = [
    "python -m venv /opt/venv",
    ". /opt/venv/bin/activate && pip install --upgrade pip setuptools wheel",
    ". /opt/venv/bin/activate && pip install --no-cache-dir --timeout=300 -r requirements.txt"
]

[start]
cmd = "/opt/venv/bin/python slack_webhook_server.py"
```

To use this, change `railway.toml` back to:
```toml
[build]
builder = "nixpacks"
```