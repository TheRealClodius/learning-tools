# Deployment Guide: Making Your Signal Agent API Available to Others

This guide walks you through deploying your Signal Agent API so others can use it.

## 🚀 Quick Deployment (Local Network)

### Step 1: Start the Server Locally
```bash
cd /Users/andreiclodius/Documents/GitHub/Agents/Signal

# Start the server (accessible to local network)
python -m uvicorn interfaces.api_interface:app --host 0.0.0.0 --port 8000 --reload
```

**Your API is now available at:**
- Local: `http://localhost:8000`
- Network: `http://YOUR-IP-ADDRESS:8000`

### Step 2: Find Your IP Address
```bash
# On macOS/Linux
ifconfig | grep "inet " | grep -v 127.0.0.1

# On Windows
ipconfig | findstr "IPv4"
```

### Step 3: Share with Others
Give others your IP address: `http://192.168.1.XXX:8000/api/send-message`

## 🔑 Step-by-Step: Setting Up API Keys for Users

### Step 1: Generate API Keys for Users
```bash
# Generate a key for each user
python generate_api_key.py generate --name "User 1 - John"
python generate_api_key.py generate --name "User 2 - Sarah"
python generate_api_key.py generate --name "Testing Team"
```

### Step 2: Share Keys Securely
For each user, provide:
1. **API Endpoint**: `http://YOUR-IP:8000/api/send-message`
2. **API Key**: `sk-signal-ABC123...` (from generation)
3. **Documentation**: Share the `SEND_MESSAGE_API.md` file

### Step 3: Test Connection
Have users test with:
```bash
curl -X POST "http://YOUR-IP:8000/api/send-message" \
  -H "Authorization: Bearer sk-signal-THEIR-KEY" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, testing my access!"}'
```

## 🌐 Production Deployment Options

### Option 1: Cloud Deployment (Railway/Render)

#### Railway Deployment (Detailed Walkthrough)

**What Railway Does**: Railway is a cloud platform that automatically builds and deploys your code. When you connect your GitHub repo, Railway will:
- Clone your code to their servers
- Detect it's a Python app (from `requirements.txt` or `pyproject.toml`)
- Install your dependencies
- Start your app on their cloud infrastructure
- Give you a public URL that anyone can access

**Step-by-Step Process:**

**1. Prepare Your Code for Deployment**
```bash
# Make sure you're in your Signal directory
cd /Users/andreiclodius/Documents/GitHub/Agents/Signal

# Add all your files to Git
git add .
git commit -m "Add Signal Agent API with rate limiting and API keys"
git push origin main
```

**2. Sign Up and Connect to Railway**
- Go to [railway.app](https://railway.app)
- Sign up with your GitHub account (this gives Railway permission to access your repos)
- Click "New Project" 
- Select "Deploy from GitHub repo"
- Choose your Signal repository from the list

**3. What Happens During Deployment (Automatic)**
Railway will automatically:
- **Detect Python**: Sees your `requirements.txt` file and knows it's a Python app
- **Install Dependencies**: Runs `pip install -r requirements.txt` on their servers
- **Build Your App**: Prepares your code to run in the cloud
- **Start the Server**: Automatically runs `uvicorn interfaces.api_interface:app --host 0.0.0.0 --port $PORT`
- **Generate URL**: Creates a public URL like `https://signal-agent-production-a1b2.up.railway.app`

**4. Configure Environment Variables (Required)**
Your app needs your Anthropic API key to work:
- In Railway dashboard, click on your project
- Go to "Variables" tab
- Add: `ANTHROPIC_API_KEY` = `your_actual_anthropic_key_here`
- Railway will automatically restart your app with the new variable

**5. Your API is Now Live!**
After deployment (usually 2-3 minutes), your API will be available at:
```
https://your-app-name-production-xyz.up.railway.app/api/send-message
```

**6. Test Your Deployed API**
```bash
# Replace with your actual Railway URL
curl -X POST "https://your-app-name-production-xyz.up.railway.app/api/send-message" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello from the cloud!"}'
```

**7. Generate API Keys for Users**
Since your app is now in the cloud, you can't run the local key generation script. Instead:

**Option A: Use the API to generate keys**
```bash
# Generate keys via your deployed API
curl -X POST "https://your-railway-url/api/admin/generate-key" \
  -H "Authorization: Bearer admin-key" \
  -H "Content-Type: application/json" \
  -d '{"name": "User 1", "prefix": "sk-signal"}'
```

**Option B: Generate locally and upload**
```bash
# Generate keys locally
python generate_api_key.py generate --name "User 1"
python generate_api_key.py generate --name "User 2"

# The api_keys.json file will need to be included in your repo
git add api_keys.json
git commit -m "Add generated API keys"
git push origin main
# Railway will automatically redeploy with the keys
```

**8. Share with Users**
Give your users:
- **API Endpoint**: `https://your-railway-url/api/send-message`
- **Their API Key**: `sk-signal-ABC123...`
- **Documentation**: The `SEND_MESSAGE_API.md` file

**Railway Dashboard Features:**
- **Logs**: See what's happening with your app in real-time
- **Metrics**: Monitor CPU, memory, and network usage
- **Deployments**: See history of all deployments
- **Custom Domain**: Add your own domain name (optional)
- **Scaling**: Automatically handles more users as needed

**Costs:**
- Railway has a generous free tier that should handle moderate usage
- You'll only pay if you exceed the free limits (unlikely for most use cases)

**What Your Users See:**
Instead of `http://192.168.1.123:8000/api/send-message` (your local IP), they use:
`https://your-app.railway.app/api/send-message` (accessible from anywhere in the world)

**Visual Overview of Railway Deployment:**
```
Your Computer                    Railway Cloud                    Your Users
--------------                   -------------                    ----------
1. Code files        ──push──>   2. Railway builds        ──>    4. Public URL
   - api_interface.py             - Installs requirements          https://your-app.railway.app
   - requirements.txt             - Starts your server    
   - generate_api_key.py          - Assigns public URL
                                                          
3. You add secrets   ──config──> Railway environment
   - ANTHROPIC_API_KEY            - Secure environment vars
```

**Common Railway Issues & Solutions:**

**Problem: "Application failed to respond"**
- **Cause**: Missing `ANTHROPIC_API_KEY` environment variable
- **Solution**: Add your Anthropic API key in Railway dashboard → Variables

**Problem: "Build failed"**  
- **Cause**: Missing or incorrect `requirements.txt`
- **Solution**: Make sure `requirements.txt` has all needed packages

**Problem: "Can't generate API keys"**
- **Cause**: `generate_api_key.py` runs locally, not on Railway
- **Solution**: Use the API endpoints to generate keys, or generate locally and commit to Git

**Problem: "Port binding failed"**
- **Cause**: Railway expects your app to use `$PORT` environment variable
- **Solution**: Railway auto-detects FastAPI and handles this, but if issues persist, update your code to: `port = int(os.environ.get("PORT", 8000))`

#### Render Deployment
1. **Create `render.yaml`**:
```yaml
services:
  - type: web
    name: signal-agent-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn interfaces.api_interface:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: ANTHROPIC_API_KEY
        sync: false
```

2. **Deploy**:
   - Connect repo to [render.com](https://render.com)
   - Your API will be at: `https://your-app.onrender.com`

### Option 2: VPS Deployment

#### DigitalOcean/AWS/GCP
1. **Create a VPS** (Ubuntu 22.04 recommended)

2. **Install Dependencies**:
```bash
# SSH into your server
ssh user@your-server-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3 python3-pip nginx -y

# Clone your repo
git clone https://github.com/your-username/Signal.git
cd Signal

# Install Python dependencies
pip3 install -r requirements.txt
```

3. **Create Service File**:
```bash
sudo nano /etc/systemd/system/signal-api.service
```

```ini
[Unit]
Description=Signal Agent API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/Signal
ExecStart=/usr/bin/python3 -m uvicorn interfaces.api_interface:app --host 0.0.0.0 --port 8000
Restart=always
Environment=ANTHROPIC_API_KEY=your_key_here

[Install]
WantedBy=multi-user.target
```

4. **Start Service**:
```bash
sudo systemctl daemon-reload
sudo systemctl enable signal-api
sudo systemctl start signal-api
```

5. **Configure Nginx**:
```bash
sudo nano /etc/nginx/sites-available/signal-api
```

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/signal-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## 🔒 Security Configuration

### Step 1: Set Up Firewall
```bash
# Ubuntu/Debian
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS
sudo ufw enable

# Block direct access to port 8000
sudo ufw deny 8000
```

### Step 2: SSL Certificate (Production)
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo systemctl enable certbot.timer
```

### Step 3: Environment Variables
```bash
# Create .env file
nano .env.local
```

```env
ANTHROPIC_API_KEY=your_anthropic_key_here
ENVIRONMENT=production
API_HOST=0.0.0.0
API_PORT=8000
```

## 👥 User Management

### Create Admin API Key
```bash
# Generate admin key
python generate_api_key.py generate --name "Admin Key" --prefix "sk-admin"

# Use this key for admin operations
export ADMIN_API_KEY="sk-admin-your-generated-key"
```

### Set Custom Limits for Users
```bash
# Set limits for specific users
curl -X POST "http://your-server/api/admin/set-limits/user-key-hash" \
  -H "Authorization: Bearer $ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "daily_limit": 50000,
    "monthly_limit": 1000000
  }'
```

### Monitor Usage
```bash
# Check all users' usage
curl -H "Authorization: Bearer $ADMIN_API_KEY" \
  "http://your-server/api/admin/all-usage"
```

## 📊 Monitoring & Maintenance

### Step 1: Set Up Logging
```bash
# Check service logs
sudo journalctl -u signal-api -f

# Check nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### Step 2: Health Monitoring
Create a simple health check script:

```bash
nano health_check.sh
```

```bash
#!/bin/bash
HEALTH_URL="http://localhost:8000/health"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $HEALTH_URL)

if [ $RESPONSE -eq 200 ]; then
    echo "✅ API is healthy"
else
    echo "❌ API is down (HTTP $RESPONSE)"
    # Restart service
    sudo systemctl restart signal-api
fi
```

```bash
chmod +x health_check.sh

# Add to crontab for regular checks
crontab -e
# Add: */5 * * * * /home/ubuntu/health_check.sh
```

### Step 3: Backup API Keys
```bash
# Backup API keys regularly
cp api_keys.json api_keys_backup_$(date +%Y%m%d).json

# Set up automatic backups
echo "0 2 * * * cp /home/ubuntu/Signal/api_keys.json /home/ubuntu/backups/api_keys_\$(date +\%Y\%m\%d).json" | crontab -
```

## 📋 User Onboarding Checklist

For each new user, provide:

### 1. API Access Information
- [ ] API Endpoint URL
- [ ] Personal API Key
- [ ] Rate limits and token allowances
- [ ] Documentation (SEND_MESSAGE_API.md)

### 2. Integration Examples
- [ ] Code examples in their preferred language
- [ ] Test curl commands
- [ ] Error handling guidance

### 3. Usage Guidelines
- [ ] Rate limiting information
- [ ] Token conservation tips
- [ ] Support contact information

### 4. Testing Checklist
- [ ] Successful API key authentication
- [ ] Basic message sending/receiving
- [ ] Rate limit handling
- [ ] Error response handling

## 🛠 Troubleshooting Common Issues

### Connection Refused
```bash
# Check if service is running
sudo systemctl status signal-api

# Check port availability
sudo netstat -tlnp | grep :8000

# Restart service
sudo systemctl restart signal-api
```

### API Key Issues
```bash
# Validate specific key
python generate_api_key.py validate sk-signal-user-key

# List all keys
python generate_api_key.py list

# Check if key validation is enabled
grep "USE_KEY_VALIDATION" interfaces/api_interface.py
```

### Rate Limiting Problems
```bash
# Check current rate limits
curl "http://your-server/api/rate-limit/status"

# Reset usage for a user
curl -X DELETE "http://your-server/api/admin/reset-usage/user-hash" \
  -H "Authorization: Bearer $ADMIN_API_KEY"
```

## 🔄 Updates and Maintenance

### Update Deployment
```bash
# Pull latest changes
git pull origin main

# Restart service
sudo systemctl restart signal-api

# Check status
sudo systemctl status signal-api
```

### Scale for More Users
1. **Increase Rate Limits**: Edit `api_rate_config` in `api_interface.py`
2. **Add Load Balancer**: Use nginx upstream for multiple instances
3. **Database Migration**: Move from JSON to PostgreSQL/MongoDB
4. **Caching**: Add Redis for token usage caching

---

## 📞 Quick Start Summary

1. **Start server**: `uvicorn interfaces.api_interface:app --host 0.0.0.0 --port 8000`
2. **Generate API keys**: `python generate_api_key.py generate --name "User Name"`
3. **Share endpoint**: `http://YOUR-IP:8000/api/send-message`
4. **Provide documentation**: Share `SEND_MESSAGE_API.md`
5. **Monitor usage**: `curl http://YOUR-IP:8000/api/admin/all-usage`

Your Signal Agent API is now ready for others to use! 🚀
