"""
Slack User Service

Manages user information, timezone handling, and database caching for Slack users.
Provides a clean interface for user data retrieval and management.
"""

import logging
import datetime
from typing import Dict, Any, Optional
from zoneinfo import ZoneInfo

# Import database for user caching
from database import user_db

logger = logging.getLogger(__name__)


class SlackUserService:
    """Service for managing Slack user information and timezones"""
    
    def __init__(self, slack_client=None):
        self.slack_client = slack_client
        
        # Timezone mapping for common labels to IANA identifiers
        self.timezone_mapping = {
            "Eastern Standard Time": "America/New_York",
            "Eastern Daylight Time": "America/New_York", 
            "Central Standard Time": "America/Chicago",
            "Central Daylight Time": "America/Chicago",
            "Mountain Standard Time": "America/Denver",
            "Mountain Daylight Time": "America/Denver",
            "Pacific Standard Time": "America/Los_Angeles",
            "Pacific Daylight Time": "America/Los_Angeles",
            "Greenwich Mean Time": "UTC",
            "Central European Time": "Europe/Paris",
            "Central European Summer Time": "Europe/Paris",
            "Eastern European Time": "Europe/Bucharest",
            "Eastern European Summer Time": "Europe/Bucharest"
        }
    
    def set_slack_client(self, slack_client):
        """Set or update the Slack client"""
        self.slack_client = slack_client
    
    async def get_user_info(self, user_id: str) -> Dict[str, Any]:
        """
        Get user information with database caching
        
        First checks database, then fetches from Slack API if not found or stale
        """
        try:
            # Check database first
            cached_user = await user_db.get_user(user_id)
            
            if cached_user:
                logger.info(f"Found user {user_id} in database cache")
                return {
                    'real_name': cached_user.get('real_name', ''),
                    'display_name': cached_user.get('display_name', ''),
                    'title': cached_user.get('title', ''),
                    'timezone': cached_user.get('timezone', 'UTC')
                }
            
            # Not in database, fetch from Slack API
            if not self.slack_client:
                logger.error("Slack client not set - cannot fetch user info")
                return self._get_default_user_info()
                
            logger.info(f"User {user_id} not in database, fetching from Slack API")
            response = await self.slack_client.users_info(user=user_id)
            
            if response["ok"]:
                user_info = response["user"]
                
                # Save to database for future use
                await user_db.save_user(user_id, user_info)
                
                # Extract key fields
                profile = user_info.get('profile', {})
                return {
                    'real_name': user_info.get('real_name', profile.get('real_name', '')),
                    'display_name': profile.get('display_name', user_info.get('name', '')),
                    'title': profile.get('title', ''),
                    'timezone': user_info.get('tz', 'UTC')
                }
            else:
                logger.error(f"Slack API error for user {user_id}: {response.get('error')}")
                return self._get_default_user_info()
                
        except Exception as e:
            logger.error(f"Error getting user info for {user_id}: {e}")
            return self._get_default_user_info()
    
    async def get_user_timezone(self, user_id: str) -> str:
        """Get user's timezone from their Slack profile"""
        try:
            if not self.slack_client:
                logger.error("Slack client not set - cannot fetch timezone")
                return "UTC"
                
            # Call Slack API to get user info
            response = await self.slack_client.users_info(user=user_id)
            
            if response["ok"]:
                user_info = response["user"]
                
                # Try to get timezone from user profile
                timezone = user_info.get("tz")
                if timezone:
                    logger.info(f"Found timezone for user {user_id}: {timezone}")
                    return timezone
                
                # Fallback: try to get from tz_label 
                tz_label = user_info.get("tz_label")
                if tz_label:
                    logger.info(f"Using tz_label for user {user_id}: {tz_label}")
                    mapped_tz = self.timezone_mapping.get(tz_label, "UTC")
                    logger.info(f"Mapped {tz_label} to {mapped_tz} for user {user_id}")
                    return mapped_tz
                    
            logger.warning(f"Could not get timezone for user {user_id}, using UTC")
            return "UTC"
            
        except Exception as e:
            logger.error(f"Error getting user timezone for {user_id}: {e}")
            return "UTC"
    
    def get_user_timestamp(self, user_timezone: str) -> str:
        """Get current timestamp in user's timezone"""
        try:
            user_tz = ZoneInfo(user_timezone)
            current_time = datetime.datetime.now(user_tz)
            return current_time.strftime("%Y-%m-%d %H:%M:%S %Z")
        except Exception as e:
            logger.warning(f"Failed to get user timezone {user_timezone}: {e}")
            # Fallback to UTC
            current_time = datetime.datetime.now(datetime.timezone.utc)
            return current_time.strftime("%Y-%m-%d %H:%M:%S UTC")
    
    def create_user_context(self, user_id: str, user_info: Dict[str, Any], 
                           channel_id: str, thread_ts: str) -> Dict[str, Any]:
        """Create a complete user context dictionary for agent processing"""
        user_timezone = user_info.get('timezone', 'UTC')
        timestamp = self.get_user_timestamp(user_timezone)
        
        return {
            "platform": "slack",
            "user_id": user_id,
            "channel_id": channel_id,
            "thread_ts": thread_ts,
            "timestamp": timestamp,
            "user_timezone": user_timezone,
            "user_name": user_info.get('real_name', ''),
            "user_display_name": user_info.get('display_name', ''),
            "user_title": user_info.get('title', '')
        }
    
    async def resolve_user_from_slack(self, user_id: str, display_name: Optional[str] = None) -> str:
        """
        Resolve a user ID to a human-readable name from Slack API
        
        Args:
            user_id: Slack user ID
            display_name: Optional display name from mention
            
        Returns:
            Human-readable name for the user
        """
        try:
            # If display name is provided in the mention, use it
            if display_name:
                logger.info(f"Using provided display name for {user_id}: {display_name}")
                return display_name
            
            if not self.slack_client:
                logger.error("Slack client not set - cannot resolve user")
                return user_id
                
            # Get user info from Slack API
            user_info = await self.slack_client.users_info(user=user_id)
            if user_info.get("ok"):
                user = user_info.get("user", {})
                # Use display name if available, otherwise real name, otherwise username
                name = (user.get("profile", {}).get("display_name") or 
                       user.get("real_name") or 
                       user.get("name", user_id))
                logger.info(f"Resolved user {user_id} to: {name}")
                return name
            else:
                logger.warning(f"Failed to get user info for {user_id}")
                return user_id
                
        except Exception as e:
            logger.error(f"Error resolving user {user_id}: {e}")
            return user_id
    
    def _get_default_user_info(self) -> Dict[str, Any]:
        """Get default user info when API calls fail"""
        return {
            'real_name': '',
            'display_name': '',
            'title': '',
            'timezone': 'UTC'
        }
