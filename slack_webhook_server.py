from fastapi import FastAPI
from interfaces.slack_interface import create_slack_app

# Create a FastAPI instance
app = FastAPI()

# Get the Slack handler
slack_handler = create_slack_app()

# Mount the Slack handler on the root path
# All requests to the root will be forwarded to the Slack app
app.mount("/", slack_handler)
