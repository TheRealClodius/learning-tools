# Signal AI Agent - Deployment Guide

This guide covers multiple deployment options for your Signal AI agent system.

## 🚀 Quick Start - Choose Your Deployment

### Option 1: **Slack Bot** (Most Common)
- Deploy `slack_webhook_server.py` 
- Requires: Slack Bot Token, Anthropic API key
- Best for: Team collaboration, automated responses

### Option 2: **Web API** 
- Deploy `interfaces/api_interface.py`
- Requires: Anthropic API key
- Best for: Web frontends, mobile apps, integrations

### Option 3: **CLI Tool**
- Run locally: `python interfaces/cli_interface.py`
- Best for: Personal use, development, scripting

---

## 🌐 Cloud Platform Deployment

### **Railway** (Recommended - Easiest)
1. Install Railway CLI: `npm install -g @railway/cli`
2. Login: `railway login` 
3. Initialize: `railway init`
4. Set environment variables in Railway dashboard
5. Deploy: `railway up`

**Start Command Options:**
- Slack Bot: `python slack_webhook_server.py`
- Web API: `python interfaces/api_interface.py`

### **Render**
1. Connect GitHub repo to Render
2. Choose "Web Service" 
3. **Build Command:** `pip install -r requirements.txt`
4. **Start Command:** `python slack_webhook_server.py`
5. Add environment variables in dashboard

### **Heroku**
```bash
heroku create your-agent-name
heroku config:set ANTHROPIC_API_KEY=your_key_here
heroku config:set SLACK_BOT_TOKEN=your_token_here
heroku config:set SLACK_SIGNING_SECRET=your_secret_here
git push heroku main
```

### **Google Cloud Run**
```bash
# Build and deploy container
gcloud run deploy signal-agent \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

## 🐳 Docker Deployment

### **Local Docker**
```bash
# Build and run Slack bot
docker build -t signal-agent .
docker run -p 8000:8000 --env-file .env.local signal-agent

# Or use docker-compose
docker-compose up signal-agent
```

### **Docker for API only**
```bash
# Run API interface instead
docker-compose --profile api up signal-api
```

---

## 🖥️ VPS/Server Deployment

### **DigitalOcean, Linode, AWS EC2, etc.**

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/Signal.git
cd Signal

# 2. Install Python 3.11+ and dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.sample .env.local  # or create .env.local manually
nano .env.local  # Add your API keys

# 4. Run with process manager
pip install gunicorn
gunicorn slack_webhook_server:app --bind 0.0.0.0:8000 --workers 2

# 5. Setup reverse proxy (nginx)
sudo apt install nginx
# Configure nginx to proxy to localhost:8000
```

### **Systemd Service (Linux)**
```bash
# Create service file
sudo nano /etc/systemd/system/signal-agent.service
```

```ini
[Unit]
Description=Signal AI Agent
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/Signal
Environment=PATH=/path/to/venv/bin
ExecStart=/path/to/venv/bin/python slack_webhook_server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable signal-agent
sudo systemctl start signal-agent
```

---

## 🔧 Environment Setup

### **Required Environment Variables**

Create `.env.local` file:
```env
# Required for all deployments
ANTHROPIC_API_KEY=your_anthropic_key_here

# Required for Slack bot
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_SIGNING_SECRET=your-signing-secret

# Optional APIs
OPENWEATHER_API_KEY=your_weather_key
PERPLEXITY_API_KEY=your_perplexity_key
GEMINI_API_KEY=your_gemini_key


```

### **Getting API Keys**

1. **Anthropic (Required)**: https://console.anthropic.com/
2. **Slack Bot**: https://api.slack.com/apps
   - Create new app → Bot Token Scopes: `app_mentions:read`, `channels:history`, `chat:write`, `im:history`, `im:read`, `users:read`
3. **OpenWeather**: https://openweathermap.org/api
4. **Perplexity**: https://www.perplexity.ai/settings/api

---

## 🔗 Slack App Configuration

### **1. Create Slack App**
- Go to https://api.slack.com/apps
- "Create New App" → "From scratch"
- Choose your workspace

### **2. Configure Event Subscriptions**
- Enable Events: **ON**
- Request URL: `https://your-domain.com/slack/events`
- Subscribe to bot events:
  - `app_mention`
  - `message.im`

### **3. Configure Interactive Components**
- Enable Interactive Components: **ON**
- Request URL: `https://your-domain.com/slack/interactive`

### **4. Configure Slash Commands**
- Create slash command: `/agent`
- Request URL: `https://your-domain.com/slack/commands`

### **5. Install App to Workspace**
- Install app → Copy Bot User OAuth Token

---

## 📊 Deployment Monitoring

### **Health Checks**
All deployments expose health endpoints:
- `GET /health` - Basic health check
- `GET /` - Service info and setup instructions

### **Logging**
Monitor logs for:
- API key validation errors
- Tool execution failures

- Slack event processing

### **Common Issues**
1. **Missing API keys** - Check environment variables
2. **Tool execution timeouts** - Check external API connectivity
3. **Slack webhook failures** - Verify signing secret and bot token

---

## 🎯 Deployment Recommendations

| Use Case | Recommended Platform | Interface | Cost |
|----------|---------------------|-----------|------|
| **Team Slack Bot** | Railway/Render | `slack_webhook_server.py` | $5-20/month |
| **Personal CLI** | Local | `cli_interface.py` | API costs only |
| **Web Integration** | Vercel/Netlify | `api_interface.py` | $0-10/month |
| **Enterprise** | AWS/GCP | Docker + Load Balancer | $50+/month |

### **Free Tier Options**
- **Render**: 750 hours/month free
- **Railway**: $5 credit monthly
- **Google Cloud Run**: 2M requests/month free
- **Heroku**: Limited free dynos

---

## 🔒 Security Considerations

1. **Never commit API keys** to git
2. **Use environment variables** for all secrets
3. **Enable HTTPS** in production
4. **Validate Slack signatures** (already implemented)
5. **Rate limit API endpoints** if needed
6. **Monitor API usage** to prevent abuse

---

## 🚀 Next Steps

1. Choose your deployment platform
2. Set up environment variables
3. Deploy the appropriate interface
4. Configure webhooks (for Slack)
5. Test with a simple message
6. Monitor logs and usage

Need help with a specific deployment? Check the platform-specific guides above or reach out!