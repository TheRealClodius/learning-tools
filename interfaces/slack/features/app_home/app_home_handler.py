"""
Slack App Home Handler

Provides a personalized home tab experience for users including:
- User dashboard with stats and recent activity
- Quick action buttons for common tasks
- Settings and preferences management
- Conversation history access
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

from database import user_db
from interfaces.slack.formatters.markdown_parser import MarkdownToSlackParser

logger = logging.getLogger(__name__)


class AppHomeHandler:
    """Handles App Home tab functionality for personalized user experiences"""
    
    def __init__(self, cache_service, user_service):
        self.cache_service = cache_service
        self.user_service = user_service
    
    async def publish_home_view(self, client, user_id: str, event: Optional[Dict] = None):
        """
        Publish the App Home tab view for a user
        
        Args:
            client: Slack web client
            user_id: User ID to publish home view for
            event: Optional event data from app_home_opened
        """
        try:
            logger.info(f"Publishing App Home view for user {user_id}")
            
            # Get user information
            user_info = await self.user_service.get_user_info(user_id)
            action_points = await self._get_action_points(user_id)
            
            # Build the home view
            home_view = await self._build_home_view(user_info, action_points)
            
            # Publish the view
            await client.views_publish(
                user_id=user_id,
                view=home_view
            )
            
            logger.info(f"Successfully published App Home view for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error publishing App Home view for user {user_id}: {e}")
            # Publish a fallback error view
            await self._publish_error_view(client, user_id)
    
    async def _build_home_view(self, user_info: Dict, action_points: List[Dict]) -> Dict:
        """Build the complete App Home view structure"""
        
        # Get user's display name and greeting
        display_name = user_info.get('display_name') or user_info.get('real_name') or 'there'
        current_hour = datetime.now().hour
        
        if current_hour < 12:
            greeting = "Good morning"
        elif current_hour < 17:
            greeting = "Good afternoon"
        else:
            greeting = "Good evening"
        
        # Build header section with integrated buttons
        header_blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{greeting}, {display_name}! 👋",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "Welcome to your personal AI agent dashboard. Here you can view your activity, manage preferences, and quickly access common features."
                },
                "accessory": {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "⚙️ Preferences",
                        "emoji": True
                    },
                    "action_id": "open_preferences"
                }
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "📝 Start New Chat",
                            "emoji": True
                        },
                        "action_id": "start_new_chat",
                        "style": "primary"
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "What I can do",
                            "emoji": True
                        },
                        "action_id": "show_available_tools"
                    }
                ]
            }
        ]
        
        # Action Points section
        action_points_blocks = await self._build_action_points_section(action_points)
        
        # Help & Support section
        help_blocks = [
            {
                "type": "divider"
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "For bug reports or feature requests, send a DM to *@Andrei Clodius*"
                    }
                ]
            }
        ]
        
        # Combine all blocks
        all_blocks = (
            header_blocks + 
            action_points_blocks + 
            help_blocks
        )
        
        return {
            "type": "home",
            "blocks": all_blocks
        }
    
    async def _build_action_points_section(self, action_points: List[Dict]) -> List[Dict]:
        """Build the action points section from recent conversations"""
        
        blocks = [
            {
                "type": "divider"
            },
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "📋 Action Points & Follow-ups",
                    "emoji": True
                }
            }
        ]
        
        if not action_points:
            blocks.extend([
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "_No recent action points found. Start chatting with me to see your personalized action items here!_"
                    }
                }
            ])
        else:
            # Create card-like sections for each action point
            for point in action_points[:3]:  # Show top 3 action points
                timestamp = point.get('timestamp', 'Recent')
                action = point.get('action', 'No action specified')
                context = point.get('context', '')
                priority = point.get('priority', 'normal')
                
                # Add priority emoji and styling
                priority_info = {
                    'high': {'emoji': '🔴', 'text': 'High Priority'},
                    'medium': {'emoji': '🟡', 'text': 'Medium Priority'}, 
                    'normal': {'emoji': '🔵', 'text': 'Normal Priority'},
                    'low': {'emoji': '⚪', 'text': 'Low Priority'}
                }.get(priority, {'emoji': '🔵', 'text': 'Normal Priority'})
                
                blocks.extend([
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"{priority_info['emoji']} *{action}*\n{context}\n\n_{priority_info['text']} • {timestamp}_"
                        }
                    },
                    {
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": " "
                            }
                        ]
                    }
                ])
            
            # Add "View All" button if there are more action points
            if len(action_points) > 3:
                blocks.append({
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": f"View All {len(action_points)} Action Points",
                                "emoji": True
                            },
                            "action_id": "view_all_action_points"
                        }
                    ]
                })
        
        return blocks
    
    async def _get_action_points(self, user_id: str) -> List[Dict]:
        """Get action points extracted from recent conversations"""
        try:
            # TODO: Implement smart action point extraction using Signal
            # This would analyze recent conversations and extract action items
            # For now, return mock data
            action_points = [
                {
                    'timestamp': '2 hours ago',
                    'action': 'Follow up on Q4 budget analysis',
                    'context': 'Based on financial data discussion in #finance',
                    'priority': 'high'
                },
                {
                    'timestamp': 'Yesterday',
                    'action': 'Review competitor research findings',
                    'context': 'Market analysis request in DM',
                    'priority': 'medium'
                },
                {
                    'timestamp': '2 days ago',
                    'action': 'Schedule team meeting about project roadmap',
                    'context': 'Planning discussion in #product',
                    'priority': 'normal'
                }
            ]
            
            return action_points
            
        except Exception as e:
            logger.error(f"Error getting action points for {user_id}: {e}")
            return []
    

    
    async def handle_app_home_action(self, ack, body, client, action_id: str):
        """Handle actions triggered from the App Home tab"""
        await ack()
        
        try:
            user_id = body.get('user', {}).get('id')
            
            if action_id == "start_new_chat":
                await self._handle_start_new_chat(body, client, user_id)
            elif action_id == "show_available_tools":
                await self._handle_show_tools(body, client, user_id)
            elif action_id == "open_preferences":
                await self._handle_open_preferences(body, client, user_id)
            elif action_id == "view_all_action_points":
                await self._handle_view_all_action_points(body, client, user_id)
            else:
                logger.warning(f"Unknown App Home action: {action_id}")
                
        except Exception as e:
            logger.error(f"Error handling App Home action {action_id}: {e}")
    
    async def _handle_start_new_chat(self, body, client, user_id: str):
        """Handle the start new chat action"""
        try:
            # Open a modal for starting a new conversation
            modal_view = {
                "type": "modal",
                "callback_id": "start_chat_modal",
                "title": {
                    "type": "plain_text",
                    "text": "Start New Chat"
                },
                "submit": {
                    "type": "plain_text",
                    "text": "Send Message"
                },
                "close": {
                    "type": "plain_text",
                    "text": "Cancel"
                },
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "What would you like to chat about? I can help with research, analysis, weather, and much more!"
                        }
                    },
                    {
                        "type": "input",
                        "block_id": "message_input",
                        "element": {
                            "type": "plain_text_input",
                            "action_id": "message_text",
                            "multiline": True,
                            "placeholder": {
                                "type": "plain_text",
                                "text": "Type your message here..."
                            }
                        },
                        "label": {
                            "type": "plain_text",
                            "text": "Your Message"
                        }
                    },
                    {
                        "type": "input",
                        "block_id": "channel_select",
                        "element": {
                            "type": "channels_select",
                            "action_id": "channel_id",
                            "placeholder": {
                                "type": "plain_text",
                                "text": "Select a channel..."
                            }
                        },
                        "label": {
                            "type": "plain_text",
                            "text": "Send to Channel"
                        },
                        "optional": True
                    }
                ]
            }
            
            await client.views_open(
                trigger_id=body["trigger_id"],
                view=modal_view
            )
            
        except Exception as e:
            logger.error(f"Error opening start chat modal: {e}")
    
    async def _handle_show_tools(self, body, client, user_id: str):
        """Handle showing what I can do in a narrative format"""
        try:
            # Create a modal with narrative examples and capabilities
            capabilities_modal = {
                "type": "modal",
                "callback_id": "capabilities_info_modal",
                "title": {
                    "type": "plain_text",
                    "text": "What I can do"
                },
                "close": {
                    "type": "plain_text",
                    "text": "Close"
                },
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "I'm your AI assistant that can help with research, analysis, and information management. Here are some things you can ask me:"
                        }
                    },
                    {
                        "type": "divider"
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*🌤️ Live weather conditions for any location*\n• \"What's the weather in Tokyo right now?\"\n• \"Will it rain in London tomorrow?\"\n• \"Show me the 5-day forecast for San Francisco\""
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*🔍 Real-time web research and analysis*\n• \"Find the latest news about renewable energy\"\n• \"Research competitor pricing for SaaS tools\"\n• \"What are the current trends in AI development?\""
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*💾 Personal knowledge base*\n• \"Remember that John prefers morning meetings\"\n• \"What did we discuss about the Q4 budget?\"\n• \"Save this document for future reference\""
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*📊 Retrieve Slack message history*\n• \"Find messages about the product launch\"\n• \"Show me conversations from last week about pricing\"\n• \"Search for files shared in #marketing channel\""
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*🤝 Team collaboration and analysis*\n• \"Summarize the key points from yesterday's meeting\"\n• \"Help me draft a project proposal\"\n• \"Analyze the feedback from our user survey\""
                        }
                    },
                    {
                        "type": "divider"
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "_Just mention me (@signal) in any channel or send me a DM with your request in natural language._"
                        }
                    }
                ]
            }
            
            await client.views_open(
                trigger_id=body["trigger_id"],
                view=capabilities_modal
            )
            
        except Exception as e:
            logger.error(f"Error showing capabilities modal: {e}")
    
    async def _handle_open_preferences(self, body, client, user_id: str):
        """Handle opening user preferences"""
        try:
            # Get current user preferences
            user_prefs = await user_db.get_user_preferences(user_id) if hasattr(user_db, 'get_user_preferences') else {}
            current_tone = user_prefs.get('tone_of_voice', 'Professional') if user_prefs else 'Professional'
            current_city = user_prefs.get('preferred_city', '') if user_prefs else ''
            
            preferences_modal = {
                "type": "modal",
                "callback_id": "user_preferences_modal",
                "title": {
                    "type": "plain_text",
                    "text": "Preferences"
                },
                "submit": {
                    "type": "plain_text",
                    "text": "Save"
                },
                "close": {
                    "type": "plain_text",
                    "text": "Cancel"
                },
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*⚙️ Agent Preferences*\nCustomize how Signal responds to you."
                        }
                    },
                    {
                        "type": "input",
                        "block_id": "tone_setting",
                        "element": {
                            "type": "static_select",
                            "action_id": "tone_select",
                            "placeholder": {
                                "type": "plain_text",
                                "text": "Select tone of voice..."
                            },
                            "initial_option": {
                                "text": {
                                    "type": "plain_text",
                                    "text": current_tone
                                },
                                "value": current_tone.lower()
                            },
                            "options": [
                                {"text": {"type": "plain_text", "text": "Professional"}, "value": "professional"},
                                {"text": {"type": "plain_text", "text": "Friendly"}, "value": "friendly"},
                                {"text": {"type": "plain_text", "text": "Casual"}, "value": "casual"},
                                {"text": {"type": "plain_text", "text": "Formal"}, "value": "formal"},
                                {"text": {"type": "plain_text", "text": "Concise"}, "value": "concise"},
                                {"text": {"type": "plain_text", "text": "Detailed"}, "value": "detailed"}
                            ]
                        },
                        "label": {
                            "type": "plain_text",
                            "text": "Tone of Voice"
                        }
                    },
                    {
                        "type": "input",
                        "block_id": "city_setting",
                        "element": {
                            "type": "plain_text_input",
                            "action_id": "city_input",
                            "placeholder": {
                                "type": "plain_text",
                                "text": "e.g., San Francisco, London, Tokyo..."
                            },
                            "initial_value": current_city
                        },
                        "label": {
                            "type": "plain_text",
                            "text": "Preferred City"
                        },
                        "hint": {
                            "type": "plain_text",
                            "text": "Used for weather, local time, and location-specific context"
                        },
                        "optional": True
                    },
                    {
                        "type": "divider"
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*🚀 Coming Soon...*\n• Pin documents and workflows as sources\n• Create custom shortcuts and commands\n• Team collaboration settings\n• Advanced context management"
                        }
                    }
                ]
            }
            
            await client.views_open(
                trigger_id=body["trigger_id"],
                view=preferences_modal
            )
            
        except Exception as e:
            logger.error(f"Error opening preferences modal: {e}")
    
    async def _handle_view_all_action_points(self, body, client, user_id: str):
        """Handle viewing all action points"""
        try:
            # Get all action points
            all_action_points = await self._get_action_points(user_id)
            
            # Create modal with action points list
            action_points_blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*📋 All Action Points & Follow-ups*"
                    }
                }
            ]
            
            if not all_action_points:
                action_points_blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "_No action points found. Start conversations with Signal to see extracted action items here!_"
                    }
                })
            else:
                for point in all_action_points:
                    timestamp = point.get('timestamp', 'Recent')
                    action = point.get('action', 'No action specified')
                    context = point.get('context', '')
                    priority = point.get('priority', 'normal')
                    
                    # Add priority emoji
                    priority_emoji = {
                        'high': '🔴',
                        'medium': '🟡', 
                        'normal': '🔵',
                        'low': '⚪'
                    }.get(priority, '🔵')
                    
                    action_points_blocks.append({
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"{priority_emoji} *{action}*\n_{context}_ • _{timestamp}_"
                        }
                    })
            
            action_points_modal = {
                "type": "modal",
                "callback_id": "all_action_points_modal",
                "title": {
                    "type": "plain_text",
                    "text": "All Action Points"
                },
                "close": {
                    "type": "plain_text",
                    "text": "Close"
                },
                "blocks": action_points_blocks
            }
            
            await client.views_open(
                trigger_id=body["trigger_id"],
                view=action_points_modal
            )
            
        except Exception as e:
            logger.error(f"Error showing all action points: {e}")
    
    async def _publish_error_view(self, client, user_id: str):
        """Publish a fallback error view when the main view fails"""
        try:
            error_view = {
                "type": "home",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "❌ *Error Loading Dashboard*\n\nSorry, there was an error loading your personal dashboard. Please try refreshing or contact support if the issue persists."
                        }
                    },
                    {
                        "type": "actions",
                        "elements": [
                            {
                                "type": "button",
                                "text": {
                                    "type": "plain_text",
                                    "text": "🔄 Refresh"
                                },
                                "action_id": "refresh_home_view"
                            }
                        ]
                    }
                ]
            }
            
            await client.views_publish(
                user_id=user_id,
                view=error_view
            )
            
        except Exception as e:
            logger.error(f"Error publishing fallback error view: {e}")
