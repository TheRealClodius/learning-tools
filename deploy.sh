#!/bin/bash

# Signal AI Agent - Deployment Helper Script
# Usage: ./deploy.sh [platform] [interface]
# Example: ./deploy.sh railway slack

set -e

PLATFORM=${1:-"railway"}
INTERFACE=${2:-"slack"}

echo "🚀 Signal AI Agent Deployment Helper"
echo "Platform: $PLATFORM"
echo "Interface: $INTERFACE"
echo ""

# Check if .env.local exists
if [ ! -f ".env.local" ]; then
    echo "⚠️  .env.local not found. Creating from sample..."
    if [ -f "memoryos-env-sample.txt" ]; then
        cp memoryos-env-sample.txt .env.local
        echo "✅ Created .env.local from sample"
        echo "⚠️  Please edit .env.local with your API keys before deploying!"
    else
        echo "❌ No sample environment file found"
        exit 1
    fi
fi

# Determine start command based on interface
case $INTERFACE in
    "slack")
        START_CMD="python slack_webhook_server.py"
        echo "📱 Deploying Slack bot interface"
        ;;
    "api")
        START_CMD="python interfaces/api_interface.py"
        echo "🌐 Deploying Web API interface"
        ;;
    "cli")
        echo "💻 CLI interface doesn't require deployment"
        echo "Run locally with: python interfaces/cli_interface.py --interactive"
        exit 0
        ;;
    *)
        echo "❌ Unknown interface: $INTERFACE"
        echo "Available: slack, api, cli"
        exit 1
        ;;
esac

# Platform-specific deployment
case $PLATFORM in
    "railway")
        echo "🚂 Deploying to Railway..."
        
        # Check if Railway CLI is installed
        if ! command -v railway &> /dev/null; then
            echo "Installing Railway CLI..."
            npm install -g @railway/cli
        fi
        
        # Check if logged in
        if ! railway whoami &> /dev/null; then
            echo "Please login to Railway:"
            railway login
        fi
        
        # Initialize if needed
        if [ ! -f "railway.json" ]; then
            railway init
        fi
        
        # Set start command
        railway variables set START_COMMAND="$START_CMD"
        
        # Deploy
        railway up
        ;;
        
    "docker")
        echo "🐳 Building and running Docker container..."
        
        # Build image
        docker build -t signal-agent .
        
        # Run container
        echo "Starting container on port 8000..."
        docker run -d \
            -p 8000:8000 \
            --env-file .env.local \
            --name signal-agent \
            signal-agent $START_CMD
            
        echo "✅ Container running at http://localhost:8000"
        ;;
        
    "docker-compose")
        echo "🐳 Using Docker Compose..."
        
        if [ "$INTERFACE" = "slack" ]; then
            docker-compose up -d signal-agent
        else
            docker-compose --profile api up -d signal-api
        fi
        ;;
        
    "heroku")
        echo "🟣 Deploying to Heroku..."
        
        # Check if Heroku CLI is installed
        if ! command -v heroku &> /dev/null; then
            echo "❌ Heroku CLI not found. Please install it first."
            exit 1
        fi
        
        # Create Procfile
        echo "web: $START_CMD" > Procfile
        
        # Initialize git if needed
        if [ ! -d ".git" ]; then
            git init
            git add .
            git commit -m "Initial commit"
        fi
        
        # Create Heroku app if needed
        if ! heroku apps:info &> /dev/null; then
            heroku create signal-agent-$(date +%s)
        fi
        
        # Set config vars from .env.local
        echo "Setting environment variables..."
        heroku config:set $(cat .env.local | grep -v '^#' | xargs)
        
        # Deploy
        git push heroku main
        ;;
        
    "local")
        echo "💻 Running locally..."
        
        # Install dependencies
        echo "Installing dependencies..."
        pip install -r requirements.txt
        
        # Run the application
        echo "Starting $INTERFACE interface..."
        python -c "
import subprocess
import os
os.environ.update(dict(line.strip().split('=', 1) for line in open('.env.local') if '=' in line and not line.startswith('#')))
subprocess.run(['$START_CMD'], shell=True)
"
        ;;
        
    *)
        echo "❌ Unknown platform: $PLATFORM"
        echo "Available platforms:"
        echo "  - railway (recommended)"
        echo "  - docker"
        echo "  - docker-compose"
        echo "  - heroku"
        echo "  - local"
        exit 1
        ;;
esac

echo ""
echo "✅ Deployment completed!"
echo ""

# Show next steps based on interface
if [ "$INTERFACE" = "slack" ]; then
    echo "🔗 Next steps for Slack bot:"
    echo "1. Configure your Slack app webhooks:"
    echo "   - Events: https://your-domain.com/slack/events"
    echo "   - Interactive: https://your-domain.com/slack/interactive" 
    echo "   - Commands: https://your-domain.com/slack/commands"
    echo "2. Test by mentioning your bot in Slack"
elif [ "$INTERFACE" = "api" ]; then
    echo "🌐 Next steps for Web API:"
    echo "1. Test the API: curl https://your-domain.com/health"
    echo "2. API documentation: https://your-domain.com/docs"
    echo "3. Integrate with your frontend application"
fi

echo ""
echo "📊 Monitor your deployment:"
echo "- Health check: https://your-domain.com/health"
echo "- Service info: https://your-domain.com/"