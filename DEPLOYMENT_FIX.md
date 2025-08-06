# Docker Build Fix for Signal AI Agent

## Problem
The Docker build was failing with this error:
```
RUN --mount=type=cache,id=s/175400fb-1281-4f46-a303-1a5c01dcbed5-/root/cache/pip,target=/root/.cache/pip python -m venv --copies /opt/venv && . /opt/venv/bin/activate && pip install -r requirements.txt 
process "/bin/bash -ol pipefail -c python -m venv --copies /opt/venv && . /opt/venv/bin/activate && pip install -r requirements.txt" did not complete successfully: exit code: 1
```

## Root Cause
The issue was caused by two problems:
1. **Missing python3-venv package**: The system didn't have the `python3.11-venv` package required to create virtual environments
2. **Complex dependencies**: Some packages like `faiss-cpu`, `sentence-transformers`, and `memoryos-pro` require extensive build dependencies

## Solutions Implemented

### Solution 1: Fixed Nixpacks Configuration (Recommended)
Updated `nixpacks.toml` to include the missing `python3.11-venv` package:

```toml
[phases.setup]
pkgs = [
    "gcc", 
    "gfortran",
    "cmake",
    "pkg-config",
    "liblapack",
    "libomp",
    "libopenblas",
    "git",
    "build-essential",
    "python3.11-venv"  # Added this package
]
```

### Solution 2: Simplified Dependencies (Fallback)
Created `nixpacks-simple.toml` that uses `requirements-simple.txt` with reduced dependencies:

**To use the simplified version:**
1. Rename `nixpacks.toml` to `nixpacks-full.toml`
2. Rename `nixpacks-simple.toml` to `nixpacks.toml`
3. Deploy

### Solution 3: Optional Memory Dependencies
Modified `tools/memory.py` to gracefully handle missing `memoryos-pro` package:
- Memory functions will return helpful error messages if the package is not available
- The application will still run without memory functionality

## Deployment Instructions

### For Railway/Nixpacks Deployment:

1. **Try the main fix first** (recommended):
   - Use the updated `nixpacks.toml` (already applied)
   - Deploy normally

2. **If that fails, use simplified version**:
   ```bash
   mv nixpacks.toml nixpacks-full.toml
   mv nixpacks-simple.toml nixpacks.toml
   git add .
   git commit -m "Use simplified dependencies for deployment"
   git push
   ```

3. **If you want full functionality later**:
   ```bash
   mv nixpacks.toml nixpacks-simple.toml
   mv nixpacks-full.toml nixpacks.toml
   git add .
   git commit -m "Restore full dependencies"
   git push
   ```

## What Each Solution Provides

### Full Dependencies (nixpacks.toml):
- ✅ Complete memory functionality with MemoryOS
- ✅ Vector search with faiss-cpu
- ✅ Sentence transformers for embeddings
- ⚠️ Longer build times
- ⚠️ More complex build requirements

### Simplified Dependencies (nixpacks-simple.toml):
- ✅ Fast deployment
- ✅ All core agent functionality
- ✅ Slack integration
- ✅ API clients (OpenAI, Anthropic, Google)
- ❌ No memory persistence
- ❌ No vector search

## Testing the Fix

You can test the fixed configuration locally:
```bash
# Test virtual environment creation
python3 -m venv --copies /tmp/test_venv
source /tmp/test_venv/bin/activate

# Test dependency installation
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt  # or requirements-simple.txt
```

## Next Steps

1. Try deploying with the main fix
2. If build times are too long or it fails, switch to simplified version
3. Monitor the deployment for any remaining issues
4. Consider gradually adding back complex dependencies if needed