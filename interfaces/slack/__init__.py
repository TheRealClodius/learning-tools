"""
Slack Integration Module

Modular Slack bot implementation with support for:
- Real-time messaging and streaming
- Advanced Block Kit formatting
- Modal interactions and workflows
- App Home and slash commands
- Caching and performance optimization
"""

from .core_slack_orchestration import SlackInterface, create_slack_app

__all__ = ['SlackInterface', 'create_slack_app']
