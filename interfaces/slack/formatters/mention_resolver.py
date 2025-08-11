"""
Slack Mention Resolver

Handles parsing and resolution of @user and #channel mentions in Slack messages.
Converts Slack's encoded mentions to readable format and back, with caching for performance.
"""

import re
import logging
from typing import Dict, Any, Tuple, Optional, Set

logger = logging.getLogger(__name__)


class SlackMentionResolver:
    """Handles mention parsing and resolution for Slack messages"""
    
    def __init__(self, cache_service, user_service):
        self.cache_service = cache_service
        self.user_service = user_service
    
    async def parse_slack_mentions(self, text: str, slack_client) -> Tuple[str, Dict[str, Any]]:
        """
        Parse incoming Slack message to clean text and extract mentions.
        
        Args:
            text: Raw Slack message text with encoded mentions
            slack_client: Slack client for API calls
            
        Returns:
            tuple: (cleaned_text, mention_context)
            mention_context: dict mapping readable names back to Slack IDs
        """
        logger.info(f"Raw message text: {repr(text)}")
        
        # Store mappings for reverse conversion
        mention_context = {
            'users': {},  # name -> user_id
            'channels': {}  # name -> channel_id
        }
        
        # Debug: Find ALL potential mentions first 
        all_mentions = re.findall(r'<[^>]+>', text)
        logger.info(f"Found all <> patterns: {all_mentions}")
        
        # Process user mentions
        text = await self._process_user_mentions(text, mention_context, slack_client)
        
        # Process channel mentions
        text = await self._process_channel_mentions(text, mention_context, slack_client)
        
        # Handle special mentions like @everyone, @here, @channel
        text = self._process_special_mentions(text)
        
        # Remove any remaining bot mentions (our own bot)
        text = re.sub(r'<@[UW][A-Z0-9]{8,}>', '', text)
        
        final_text = ' '.join(text.split()).strip()
        logger.info(f"Final cleaned text: {repr(final_text)}")
        logger.info(f"Mention context: {mention_context}")
        
        return final_text, mention_context
    
    async def _process_user_mentions(self, text: str, mention_context: Dict[str, Any], slack_client) -> str:
        """Process user mentions in the text"""
        # Find all user mentions - more flexible pattern to handle different formats
        # Slack can send <@U123456789>, <@WLZK08431>, or other variants - not all start with U!
        user_mentions = re.findall(r'<@([A-Z0-9]+)(?:\|([^>]+))?>', text)
        logger.info(f"Found user mentions: {user_mentions}")
        
        # Additional debug: try broader pattern
        broad_user_mentions = re.findall(r'<@([^>|]+)(?:\|([^>]+))?>', text)
        logger.info(f"Broader user mention pattern found: {broad_user_mentions}")
        
        for user_id, display_name in user_mentions:
            try:
                name = await self.user_service.resolve_user_from_slack(user_id, display_name)
                
                # Store mapping for reverse conversion
                mention_context['users'][name] = user_id
                
                # Replace the mention with readable name
                mention_pattern = f'<@{user_id}' + (f'|{re.escape(display_name)}' if display_name else '') + '>'
                text = text.replace(mention_pattern, f'@{name}')
                
            except Exception as e:
                logger.error(f"Error resolving user {user_id}: {e}")
                # Fallback: just show @user_id
                name = user_id
                mention_context['users'][name] = user_id
                mention_pattern = f'<@{user_id}' + (f'|{re.escape(display_name)}' if display_name else '') + '>'
                text = text.replace(mention_pattern, f'@{user_id}')
        
        return text
    
    async def _process_channel_mentions(self, text: str, mention_context: Dict[str, Any], slack_client) -> str:
        """Process channel mentions in the text"""
        # Find all channel mentions <#C123456789|channel-name> or <#C123456789>
        channel_mentions = re.findall(r'<#(C[A-Z0-9]+)(?:\|([^>]+))?>', text)
        logger.info(f"Found channel mentions: {channel_mentions}")
        
        for channel_id, channel_name in channel_mentions:
            try:
                logger.info(f"Processing channel mention: channel_id={channel_id}, channel_name={channel_name}")
                
                if channel_name:
                    # Channel name is already provided in the mention
                    name = channel_name
                    logger.info(f"Using provided channel name for {channel_id}: {name}")
                else:
                    # Get channel info from Slack API
                    channel_info = await slack_client.conversations_info(channel=channel_id)
                    if channel_info.get("ok"):
                        channel = channel_info.get("channel", {})
                        name = channel.get("name", channel_id)
                        logger.info(f"Resolved channel {channel_id} to: {name}")
                    else:
                        name = channel_id
                        logger.warning(f"Failed to get channel info for {channel_id}")
                
                # Store mapping for reverse conversion with the original channel name if available
                display_name = channel_name if channel_name else name
                mention_context['channels'][display_name] = channel_id
                
                # Replace the mention with readable name
                mention_pattern = f'<#{channel_id}' + (f'|{re.escape(channel_name)}' if channel_name else '') + '>'
                logger.info(f"Trying to replace pattern: '{mention_pattern}' with '#{display_name}'")
                logger.info(f"Text before replacement: {repr(text)}")
                
                original_text = text
                text = text.replace(mention_pattern, f'#{display_name}')
                
                if text == original_text:
                    logger.warning(f"Channel mention replacement failed! Pattern '{mention_pattern}' not found in text")
                else:
                    logger.info(f"Successfully replaced channel mention")
                
                logger.info(f"Text after replacement: {repr(text)}")
                
            except Exception as e:
                logger.error(f"Error resolving channel {channel_id}: {e}")
                # Fallback: just show #channel_id
                fallback_name = channel_name if channel_name else channel_id
                mention_context['channels'][fallback_name] = channel_id
                mention_pattern = f'<#{channel_id}' + (f'|{re.escape(channel_name)}' if channel_name else '') + '>'
                text = text.replace(mention_pattern, f'#{fallback_name}')
        
        return text
    
    def _process_special_mentions(self, text: str) -> str:
        """Process special mentions like @everyone, @here, @channel"""
        text = re.sub(r'<!everyone>', '@everyone', text)
        text = re.sub(r'<!here>', '@here', text)
        text = re.sub(r'<!channel>', '@channel', text)
        return text
    
    async def resolve_and_format_mentions(self, text: str, mention_context: Dict[str, Any], slack_client) -> str:
        """
        Resolve and format @user and #channel mentions in agent responses.
        
        This function first uses the initial `mention_context` as a cache.
        If a mention is not found in the cache, it will perform a live lookup.
        
        Args:
            text: Text with @user and #channel mentions to resolve
            mention_context: Context from original message parsing
            slack_client: Slack client for API calls
            
        Returns:
            Text with mentions converted to Slack format
        """
        # --- Step 1: Handle User Mentions ---
        text = await self._resolve_user_mentions(text, mention_context, slack_client)
        
        # --- Step 2: Handle Channel Mentions ---
        text = await self._resolve_channel_mentions(text, mention_context, slack_client)
        
        logger.info(f"Resolved mentions back to Slack format: {repr(text)}")
        return text
    
    async def _resolve_user_mentions(self, text: str, mention_context: Dict[str, Any], slack_client) -> str:
        """Resolve @user mentions in text"""
        # Find all potential user mentions, e.g., @andrei, @andrei.clodius, @andrei-clodius
        user_mention_pattern = r'@([a-zA-Z0-9._-]+)'
        
        # Create a list of all unique user names mentioned
        mentioned_user_names = set(re.findall(user_mention_pattern, text))
        
        # First, check the cache from the original message
        cached_users = mention_context.get('users', {})
        
        for name in mentioned_user_names:
            if name in cached_users:
                # This user was in the original message, use the cached ID
                user_id = cached_users[name]
                text = text.replace(f'@{name}', f'<@{user_id}>')
            else:
                # Live lookup needed for this user
                user_id = await self.cache_service.find_user_id(name, slack_client)
                if user_id:
                    text = text.replace(f'@{name}', f'<@{user_id}>')
                else:
                    logger.warning(f"Could not resolve user: @{name}")
        
        return text
    
    async def _resolve_channel_mentions(self, text: str, mention_context: Dict[str, Any], slack_client) -> str:
        """Resolve #channel mentions in text"""
        # Find all potential channel mentions, e.g., #general, #team-updates
        channel_mention_pattern = r'#([a-zA-Z0-9_-]+)'
        
        # Create a list of all unique channel names mentioned
        mentioned_channel_names = set(re.findall(channel_mention_pattern, text))
        
        # First, check the cache from the original message
        cached_channels = mention_context.get('channels', {})
        
        for name in mentioned_channel_names:
            if name in cached_channels:
                # This channel was in the original message, use the cached ID
                channel_id = cached_channels[name]
                text = text.replace(f'#{name}', f'<#{channel_id}|{name}>')
            else:
                # Live lookup needed for this channel
                channel_id = await self.cache_service.find_channel_id(name, slack_client)
                if channel_id:
                    text = text.replace(f'#{name}', f'<#{channel_id}|{name}>')
                else:
                    logger.warning(f"Could not resolve channel: #{name}")
        
        return text
