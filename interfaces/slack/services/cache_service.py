"""
Slack Cache Service

Manages various caching mechanisms for Slack operations:
- Execution details persistence in database for modal interactions
- User and channel ID resolution caching
- Automatic cache cleanup and TTL management
"""

import time
import asyncio
import logging
from typing import Dict, Any, Optional, Tuple, List

# Import database for persistent execution details storage
from database import user_db

logger = logging.getLogger(__name__)


class SlackCacheService:
    """Centralized caching service for Slack operations"""
    
    def __init__(self, cache_ttl: int = 3600):
        # Execution details are now stored persistently in database (no in-memory cache)
        # Database handles TTL automatically with 7-day expiration
        self.cache_ttl = cache_ttl  # Keep for user/channel cache TTL
        
        # User and channel lookup caches (still in-memory for speed)
        self.user_cache: Dict[str, str] = {}  # { 'user_name': 'USER_ID' }
        self.channel_cache: Dict[str, str] = {}  # { 'channel_name': 'CHANNEL_ID' }
        
        # Background cleanup task tracking (now for database cleanup)
        self._cleanup_task_started = False
    
    async def ensure_cleanup_task_started(self):
        """Start the database cleanup task if not already started"""
        if not self._cleanup_task_started:
            self._cleanup_task_started = True
            asyncio.create_task(self._cleanup_database_periodically())
    
    # === Execution Details Database Storage ===
    
    async def store_execution_details(self, message_ts: str, channel_id: str, user_id: str, execution_summary: List) -> None:
        """Store execution details persistently in database"""
        try:
            await user_db.save_execution_details(message_ts, channel_id, user_id, execution_summary)
            logger.info(f"DB-STORED: Execution details saved for message_ts={message_ts}, "
                       f"details_count={len(execution_summary)}")
        except Exception as e:
            logger.error(f"Failed to store execution details in database: {e}")
    
    async def get_execution_details(self, message_ts: str) -> Optional[Tuple[float, List]]:
        """Get execution details from database"""
        try:
            execution_data = await user_db.get_execution_details(message_ts)
            
            if not execution_data:
                logger.warning(f"No execution details found in database for message_ts: {message_ts}")
                return None
            
            # Return in same format as before (timestamp, data) for compatibility
            # Use current time as timestamp since we don't need it for staleness checks anymore
            current_time = time.time()
            logger.info(f"DB-HIT: Retrieved {len(execution_data)} execution details from database")
            return (current_time, execution_data)
            
        except Exception as e:
            logger.error(f"Failed to retrieve execution details from database: {e}")
            return None
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics (execution details now in database)"""
        try:
            # Could query database for execution details count if needed
            return {
                "execution_details_storage": "database",
                "user_cache_count": len(self.user_cache),
                "channel_cache_count": len(self.channel_cache),
                "cache_ttl": self.cache_ttl
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {
                "execution_details_storage": "database",
                "user_cache_count": len(self.user_cache),
                "channel_cache_count": len(self.channel_cache),
                "cache_ttl": self.cache_ttl,
                "error": str(e)
            }
    
    # === User Cache ===
    
    async def find_user_id(self, user_name: str, slack_client) -> Optional[str]:
        """Find a user ID by name, using cache to reduce API calls"""
        # Check cache first
        if user_name in self.user_cache:
            return self.user_cache[user_name]
        
        # If not in cache, fetch all users and populate cache
        try:
            logger.info("User cache empty. Fetching all users from Slack.")
            async for page in await slack_client.users_list(limit=1000):
                for user in page.get("members", []):
                    if user.get("deleted") or user.get("is_bot"):
                        continue
                    
                    # Cache by display name, real name, and username
                    display_name = user.get("profile", {}).get("display_name", "").lower()
                    real_name = user.get("real_name", "").lower()
                    name = user.get("name", "").lower()
                    user_id = user.get("id")
                    
                    if display_name: 
                        self.user_cache[display_name] = user_id
                    if real_name: 
                        self.user_cache[real_name] = user_id
                    if name: 
                        self.user_cache[name] = user_id
            
            logger.info(f"User cache populated with {len(self.user_cache)} users.")
            
            # Try to find the user again from the populated cache
            return self.user_cache.get(user_name.lower())
            
        except Exception as e:
            logger.error(f"Error fetching users from Slack: {e}")
            return None
    
    # === Channel Cache ===
    
    async def find_channel_id(self, channel_name: str, slack_client) -> Optional[str]:
        """Find a channel ID by name, using cache"""
        # Check cache first
        if channel_name in self.channel_cache:
            return self.channel_cache[channel_name]
            
        # If not in cache, fetch all public channels and populate cache
        try:
            logger.info("Channel cache empty. Fetching all public channels from Slack.")
            async for page in await slack_client.conversations_list(
                limit=1000, 
                types="public_channel"
            ):
                for channel in page.get("channels", []):
                    name = channel.get("name", "").lower()
                    channel_id = channel.get("id")
                    if name and channel_id:
                        self.channel_cache[name] = channel_id
            
            logger.info(f"Channel cache populated with {len(self.channel_cache)} channels.")

            # Try to find the channel again from the populated cache
            return self.channel_cache.get(channel_name.lower())

        except Exception as e:
            logger.error(f"Error fetching channels from Slack: {e}")
            return None
    
    # === Database Cleanup ===
    
    async def _cleanup_database_periodically(self):
        """Background task to clean up expired execution details from database"""
        while True:
            try:
                deleted_count = await user_db.cleanup_expired_executions()
                if deleted_count > 0:
                    logger.info(f"Database cleanup: {deleted_count} expired execution details removed")
            except Exception as e:
                logger.error(f"Database cleanup error: {e}")
            
            # Check every 12 hours for database cleanup
            await asyncio.sleep(12 * 3600)
    
    def clear_all_caches(self):
        """Clear in-memory caches (execution details are in database)"""
        self.user_cache.clear()
        self.channel_cache.clear()
        logger.info("In-memory caches cleared (execution details persist in database)")
