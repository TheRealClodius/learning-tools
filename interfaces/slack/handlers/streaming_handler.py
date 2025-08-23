"""
Slack Streaming Handler

Manages real-time message updates during agent execution with a clean architecture:
- Displays progressive thinking content from Claude
- Shows tool execution with pre-generated Gemini summaries from ClientAgent  
- Provides interactive execution details modal support
- Focuses purely on UI formatting (no LLM logic)
"""

import time
import logging
from typing import Dict, Any, Optional, List

from ..formatters.markdown_parser import MarkdownToSlackParser

logger = logging.getLogger(__name__)


class SlackStreamingHandler:
    """
    Handles progressive message updates for agent execution streaming.
    
    Architecture:
    - Reasoning blocks: Thinking content from Claude
    - Tool blocks: Pre-generated summaries from ClientAgent (Gemini)
    - Pure UI formatting: No LLM logic in this class
    """
    
    def __init__(self, app_client, channel_id: str, thread_ts: str, user_id: str = None):
        self.app_client = app_client
        self.channel_id = channel_id
        self.thread_ts = thread_ts
        self.user_id = user_id  # Store user_id for database storage
        self.message_ts: Optional[str] = None
        self.content_blocks = []  # Chronological list of thinking and tool blocks
        self.current_tool_block = None  # Currently active tool block
        self.all_thinking = []  # All thinking content accumulated
        self.execution_summary = []  # Store execution details for modal
        # Summarization is now handled by ClientAgent
        
    async def start_streaming(self):
        """Post initial thinking message"""
        response = await self.app_client.chat_postMessage(
            channel=self.channel_id,
            blocks=[
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": "_Reasoning..._"
                        }
                    ]
                }
            ],
            text="Reasoning...",
            thread_ts=self.thread_ts
        )
        self.message_ts = response["ts"]
        
    async def update_thinking(self, thinking_block: str):
        """Add thinking content block and update display efficiently"""
        if not self.message_ts or not thinking_block.strip():
            return
            
        # Split thinking block into lines for display formatting
        lines = thinking_block.strip().split('\n')
        clean_lines = [line.strip() for line in lines if line.strip()]
        
        # Add all lines from this thinking block
        self.all_thinking.extend(clean_lines)
        
        # DEBUG: Log thinking collection
        logger.info(f"DEBUG-THINKING: Added {len(clean_lines)} thinking lines, total entries: {len(self.all_thinking)}")
        
        await self._update_display()
    
    async def start_tool(self, tool_name: str, tool_args: Dict = None):
        """Start a new tool execution block"""
        # End any current thinking block by moving it to completed blocks
        if self.all_thinking:
            thinking_text = "\n".join(self.all_thinking)
            # Add to both content_blocks (for real-time display) and execution_summary (for modal)
            self.content_blocks.append(("thinking", thinking_text))
            self.execution_summary.append(("thinking", thinking_text))
            logger.info(f"DEBUG-THINKING: Tool start - stored {len(self.all_thinking)} thinking lines in both content_blocks and execution_summary")
            self.all_thinking = []
            
            # Note: Cache updates are handled by finish_with_response() method
        
        # Store any previous tool in execution summary
        if self.current_tool_block:
            self.execution_summary.append(("tool", self.current_tool_block.copy()))
        
        # DEBUG: Log tool start
        logger.debug(f"DEBUG: Starting tool {tool_name}, execution_summary now has {len(self.execution_summary)} items")
        
        # Extract clean tool name for display
        display_name = tool_name
        if isinstance(tool_name, dict) and "name" in tool_name:
            display_name = tool_name["name"]
        elif isinstance(tool_name, str) and tool_name.startswith("⚡️"):
            display_name = tool_name[2:]  # Remove ⚡️ prefix
        
        # Start new tool block with structured data
        self.current_tool_block = {
            "name": display_name,
            "args": tool_args or {},
            "status": "running",
            "operations": [],
            "error": None,
            "result": None
        }
        await self._update_display()
        
    # Removed append_to_current_tool - we now use structured data instead of streaming chunks
    async def complete_tool(self, result_data: any = None, error: str = None):
        """Complete current tool with structured result data and error information"""
        if not self.current_tool_block:
            return
            
        self.current_tool_block["status"] = "completed" if not error else "failed"
        
        # Store structured result data instead of Gemini summary
        if result_data:
            self.current_tool_block["result"] = result_data
        if error:
            self.current_tool_block["error"] = error
        
        # Store completed tool for display and modal
        completed_tool = self.current_tool_block.copy()
        self.content_blocks.append(("tool", completed_tool))
        self.execution_summary.append(("tool", completed_tool))
        
        # Note: Cache updates are handled by finish_with_response() method
        
        # Reset for next tool
        self.current_tool_block = None
        await self._update_display()
    
    def _format_tool_block(self, tool_info: Dict) -> str:
        """Format tool information programmatically for clean display"""
        tool_name = tool_info.get('name', 'Unknown tool')
        status = tool_info.get('status', 'running')
        error = tool_info.get('error')
        result = tool_info.get('result')
        tool_args = tool_info.get('args', {})
        
        # Start with tool name and operation
        parts = []
        
        if status == "running":
            parts.append(f"⚡️ *{tool_name}* executing...")
        elif status == "failed" and error:
            parts.append(f"⚡️ *{tool_name}* failed: {error}")
        elif status == "completed":
            # Format success message based on tool type and result
            success_msg = self._format_tool_success(tool_name, tool_args, result)
            parts.append(success_msg)
        else:
            parts.append(f"⚡️ *{tool_name}* completed")
        
        # Combine all parts and italicize the entire block
        tool_content = '\n'.join(parts)
        return f"_{tool_content}_"
    
    def _format_tool_success(self, tool_name: str, tool_args: Dict, result: any) -> str:
        """Format successful tool execution message based on tool type and result data"""
        try:
            # Handle different tool types with specific formatting
            if "weather" in tool_name.lower():
                return self._format_weather_success(tool_name, tool_args, result)
            elif "perplexity" in tool_name.lower():
                return self._format_perplexity_success(tool_name, tool_args, result)
            elif "registry" in tool_name.lower() or "reg." in tool_name.lower():
                return self._format_registry_success(tool_name, tool_args, result)
            elif "slack" in tool_name.lower():
                return self._format_slack_success(tool_name, tool_args, result)
            elif "memory" in tool_name.lower():
                return self._format_memory_success(tool_name, tool_args, result)
            else:
                # Generic success format
                return f"⚡️ *{tool_name}* completed successfully"
        except Exception as e:
            logger.warning(f"Tool formatting error for {tool_name}: {e}")
            return f"⚡️ *{tool_name}* completed"
    
    def _format_weather_success(self, tool_name: str, tool_args: Dict, result: any) -> str:
        """Format weather tool success messages"""
        if isinstance(result, dict) and result.get("success"):
            data = result.get("data", {})
            if "current" in tool_name:
                # Current weather
                temp = data.get("main", {}).get("temp")
                weather_desc = data.get("weather", [{}])[0].get("description", "")
                city = data.get("name", tool_args.get("location", "location"))
                if temp and weather_desc:
                    return f"⚡️ *{tool_name}* got weather for {city}. Currently {temp}°C, {weather_desc}"
                elif temp:
                    return f"⚡️ *{tool_name}* got weather for {city}. Currently {temp}°C"
            elif "search" in tool_name:
                # Location search
                if isinstance(data, list) and data:
                    location = data[0]
                    name = location.get("name", "location")
                    country = location.get("country", "")
                    return f"⚡️ *{tool_name}* found location: {name}, {country}"
        return f"⚡️ *{tool_name}* retrieved weather data"
    
    def _format_perplexity_success(self, tool_name: str, tool_args: Dict, result: any) -> str:
        """Format Perplexity tool success messages"""
        query = tool_args.get("query", "query")
        if isinstance(result, dict):
            if result.get("success"):
                return f"⚡️ *{tool_name}* researched: {query}"
            else:
                error_msg = result.get("message", "API error")
                return f"⚡️ *{tool_name}* failed: {error_msg}"
        return f"⚡️ *{tool_name}* completed research"
    
    def _format_registry_success(self, tool_name: str, tool_args: Dict, result: any) -> str:
        """Format registry tool success messages"""
        if "search" in tool_name:
            query = tool_args.get("query", "tools")
            return f"⚡️ *{tool_name}* searched for \"{query}\" in registry"
        elif "describe" in tool_name:
            tool_to_describe = tool_args.get("tool_name", "tool")
            return f"⚡️ *{tool_name}* verified {tool_to_describe} description"
        elif "list" in tool_name:
            return f"⚡️ *{tool_name}* listed available tools"
        elif "categories" in tool_name:
            return f"⚡️ *{tool_name}* retrieved tool categories"
        return f"⚡️ *{tool_name}* completed registry operation"
    
    def _format_slack_success(self, tool_name: str, tool_args: Dict, result: any) -> str:
        """Format Slack tool success messages"""
        if isinstance(result, dict) and result.get("success"):
            if "search" in tool_name or "vector_search" in tool_name:
                # Parse the results text to extract message count
                results_text = result.get("results", "")
                if "Found" in results_text and "messages" in results_text:
                    # Extract count from text like "Found 3 messages matching..."
                    import re
                    match = re.search(r'Found (\d+) messages?', results_text)
                    if match:
                        count = int(match.group(1))
                        return f"⚡️ *{tool_name}* found {count} messages"
                # Fallback: check if we have results content
                if results_text and len(results_text.strip()) > 50:
                    return f"⚡️ *{tool_name}* found messages"
                else:
                    return f"⚡️ *{tool_name}* found 0 messages"
            elif "channels" in tool_name:
                # For channels, try to parse from results or check data
                data = result.get("data", {})
                channels_data = data.get("channels", [])
                if channels_data:
                    count = len(channels_data)
                    return f"⚡️ *{tool_name}* retrieved {count} channels"
                # Try parsing from results text
                results_text = result.get("channels", "")
                if results_text and len(results_text.strip()) > 10:
                    return f"⚡️ *{tool_name}* retrieved channels"
        return f"⚡️ *{tool_name}* completed Slack operation"
    
    def _format_memory_success(self, tool_name: str, tool_args: Dict, result: any) -> str:
        """Format memory tool success messages"""
        if "retrieve" in tool_name:
            query = tool_args.get("query", "information")
            return f"⚡️ *{tool_name}* searched for \"{query}\""
        elif "add" in tool_name:
            return f"⚡️ *{tool_name}* stored information"
        return f"⚡️ *{tool_name}* completed memory operation"
    
    async def _update_display(self):
        """Update the streaming message display with chronological thinking and tool blocks"""
        logger.info(f"🎨 DISPLAY-UPDATE: Starting display update, message_ts={self.message_ts}")
        
        if not self.message_ts:
            logger.warning(f"🎨 DISPLAY-UPDATE: No message_ts - skipping update")
            return
            
        blocks = []
        
        # Add all completed content blocks chronologically
        for block_type, content in self.content_blocks:
            if block_type == "thinking":
                # Thinking block: small font, italics - handle multi-line properly
                thinking_lines = content.split('\n')
                thinking_formatted = '\n'.join([f"_{line.strip()}_" for line in thinking_lines if line.strip()])
                # Ensure text is never empty (Slack requires at least 1 character)
                if not thinking_formatted:
                    thinking_formatted = "_Received an empty response_"
                blocks.append({
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": thinking_formatted
                        }
                    ]
                })
            elif block_type == "tool":
                # Tool block: Format structured tool information programmatically
                tool_info = content
                tool_formatted = self._format_tool_block(tool_info)
                
                # Ensure tool_formatted is never empty
                if not tool_formatted or not tool_formatted.strip():
                    tool_formatted = "_⚡️ Tool completed_"
                
                blocks.append({
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": tool_formatted
                        }
                    ]
                })
        
        # Add current thinking block if we have ongoing thinking
        if self.all_thinking:
            thinking_lines = self.all_thinking
            thinking_formatted = '\n'.join([f"_{line.strip()}_" for line in thinking_lines if line.strip()])
            # Ensure text is never empty (Slack requires at least 1 character)
            if not thinking_formatted:
                thinking_formatted = "_Processing..._"
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": thinking_formatted
                    }
                ]
            })
        
        # Add current tool block if we have one running
        if self.current_tool_block:
            tool_formatted = self._format_tool_block(self.current_tool_block)
            
            # Ensure tool_formatted is never empty
            if not tool_formatted or not tool_formatted.strip():
                tool_formatted = "_⚡️ Tool executing..._"
            
            blocks.append({
                "type": "context", 
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": tool_formatted
                    }
                ]
            })
        
        # If no blocks yet, show initial message
        if not blocks:
            blocks = [{
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn", 
                        "text": "_Reasoning..._"
                    }
                ]
            }]
        
        # 🔍 PRODUCTION DEBUG: Log what's being sent to Slack
        logger.info(f"🎨 DISPLAY-UPDATE: Sending to Slack - blocks_count={len(blocks)} content_blocks={len(self.content_blocks)} current_tool_operations={len(self.current_tool_block.get('operations', []) if self.current_tool_block else [])}")
        
        # Log block content for debugging
        for i, block in enumerate(blocks):
            if block.get("type") == "context" and block.get("elements"):
                text_content = block["elements"][0].get("text", "")[:100]
                logger.info(f"🎨 DISPLAY-UPDATE: Block {i}: {text_content}...")
        
        try:
            await self.app_client.chat_update(
                channel=self.channel_id,
                ts=self.message_ts,
                blocks=blocks,
                text="Agent working..."
            )
            logger.info(f"🎨 DISPLAY-UPDATE: Successfully updated Slack message")
        except Exception as e:
            logger.error(f"🎨 DISPLAY-UPDATE ERROR: Failed to update Slack message: {e}")
            import traceback
            logger.error(f"🎨 DISPLAY-UPDATE TRACEBACK: {traceback.format_exc()}")
        
    async def finish_with_response(self, final_response: str, slack_interface=None):
        """Replace thinking message with final response and add execution details button"""
        if not self.message_ts:
            return
        
        # Store any remaining thinking in both content_blocks and execution summary
        if self.all_thinking:
            thinking_text = "\n".join(self.all_thinking)
            # Add to both for consistency (though final response replaces the display anyway)
            self.content_blocks.append(("thinking", thinking_text))
            self.execution_summary.append(("thinking", thinking_text))
            logger.info(f"DEBUG-THINKING: finish_with_response - stored {len(self.all_thinking)} thinking lines in both content_blocks and execution_summary")
            self.all_thinking = []
        
        # Store any current tool in execution summary
        if self.current_tool_block:
            self.execution_summary.append(("tool", self.current_tool_block.copy()))
        
        # DEBUG: Log execution summary state
        logger.info(f"DEBUG: finish_with_response - execution_summary has {len(self.execution_summary)} items")
        if self.execution_summary:
            logger.info(f"DEBUG: execution_summary types: {[item[0] for item in self.execution_summary[:5]]}")
        
        # IMPROVED: Store execution details in database for persistent access
        if slack_interface and self.execution_summary:
            try:
                # Store in database via cache service
                await slack_interface.cache_service.store_execution_details(
                    self.message_ts, self.channel_id, self.user_id or 'unknown', self.execution_summary.copy()
                )
            except Exception as e:
                logger.error(f"Failed to store execution details in database: {e}")
        else:
            logger.warning(f"DEBUG: Not storing execution details - slack_interface={slack_interface is not None}, execution_summary={len(self.execution_summary) if self.execution_summary else 0}")
        
        # Parse the response as markdown and convert to Slack blocks
        try:
            response_blocks = MarkdownToSlackParser.parse_to_blocks(final_response)
            if not response_blocks:
                response_blocks = [{
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": final_response
                    }
                }]
            
            # Add discrete execution details link if we have execution data
            if self.execution_summary:
                logger.info(f"DEBUG: Adding view flow button for message_ts={self.message_ts}")
                # Use small gray button (no "primary" style = default gray)
                execution_button_block = {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "view flow"
                            },
                            "action_id": "view_execution_details",
                            "value": self.message_ts
                        }
                    ]
                }
                response_blocks.append(execution_button_block)
            else:
                logger.info("DEBUG: No execution_summary, not adding view flow button")
            
            await self.app_client.chat_update(
                channel=self.channel_id,
                ts=self.message_ts,
                blocks=response_blocks,
                text=final_response  # Fallback for notifications
            )
            
        except Exception as e:
            logger.error(f"Error parsing markdown response: {e}")
            # Fallback to simple text block
            await self.app_client.chat_update(
                channel=self.channel_id,
                ts=self.message_ts,
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": final_response or "Error displaying response"
                        }
                    }
                ],
                text=final_response
            )
    
    def _format_tool_block(self, tool_info: Dict[str, Any]) -> str:
        """Format tool information into a readable Slack message"""
        if not tool_info:
            return "_⚡️ Tool executed_"
        
        # Extract tool information
        tool_name = tool_info.get('tool', 'unknown_tool')
        success = tool_info.get('success', True)
        summary = tool_info.get('summary', '')
        
        # Format based on success
        if success:
            status_emoji = "⚡️"
            status_text = "completed successfully"
        else:
            status_emoji = "❌"
            status_text = "encountered an error"
        
        # Build formatted message
        formatted_parts = [f"_{status_emoji} *{tool_name}* {status_text}_"]
        
        if summary and summary.strip():
            formatted_parts.append(f"_{summary.strip()}_")
        
        return "\n".join(formatted_parts)
