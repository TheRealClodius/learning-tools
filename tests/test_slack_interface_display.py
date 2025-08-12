#!/usr/bin/env python3
"""
Slack Interface Display Testing

Comprehensive tests for Slack interface information display to ensure
proper formatting, message structure, and visual presentation.

Tests cover:
- Message formatting and Block Kit structure
- Markdown to Slack conversion
- Streaming handler display updates
- Error message presentation
- Modal interactions
- Mention resolution

Usage:
    python3 tests/test_slack_interface_display.py
"""

import asyncio
import time
import json
import sys
import os
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from interfaces.slack.core_slack_orchestration import SlackInterface
from interfaces.slack.handlers.streaming_handler import SlackStreamingHandler
from interfaces.slack.formatters.markdown_parser import MarkdownToSlackParser
from interfaces.slack.formatters.mention_resolver import SlackMentionResolver
from interfaces.slack.services.cache_service import SlackCacheService
from interfaces.slack.services.user_service import SlackUserService
from interfaces.slack.handlers.modal_handler import SlackModalHandler


class SlackInterfaceDisplayTest:
    """Test Slack interface display functionality"""
    
    def __init__(self):
        self.user_id = f"display_test_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        self.test_results = []
        self.mock_client = None
        self.mock_app = None
        
    async def setup(self):
        """Initialize test environment with mocked Slack components"""
        try:
            # Create mock Slack client and app
            self.mock_client = AsyncMock()
            self.mock_app = MagicMock()
            self.mock_app.client = self.mock_client
            
            # Mock successful API responses
            self.mock_client.chat_postMessage.return_value = {"ts": "1234567890.123456", "ok": True}
            self.mock_client.chat_update.return_value = {"ok": True}
            self.mock_client.users_info.return_value = {
                "ok": True,
                "user": {
                    "id": self.user_id,
                    "name": "testuser",
                    "real_name": "Test User",
                    "tz": "America/New_York",
                    "profile": {
                        "display_name": "Test User",
                        "real_name": "Test User",
                        "title": "Test Engineer"
                    }
                }
            }
            
            print("✅ Mock Slack environment initialized")
            return True
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False

    async def test_markdown_to_slack_conversion(self):
        """Test markdown to Slack Block Kit conversion"""
        print(f"\n🔧 Test: Markdown to Slack Block Kit Conversion")
        
        test_cases = [
            {
                "name": "Simple text",
                "input": "Hello world!",
                "expected_blocks": 1,
                "expected_type": "section"
            },
            {
                "name": "Headers",
                "input": "# Main Header\n## Sub Header\nContent",
                "expected_blocks": 3,
                "expected_types": ["header", "section", "section"]
            },
            {
                "name": "Code blocks",
                "input": "```python\nprint('hello')\n```",
                "expected_blocks": 1,
                "expected_content": "```"
            },
            {
                "name": "Lists and formatting",
                "input": "**Bold text**\n- Item 1\n- Item 2\n*Italic*",
                "expected_blocks": 1,
                "expected_content": "*Bold text*"
            },
            {
                "name": "Long content splitting",
                "input": "A" * 4000,  # Exceeds Slack's limit
                "expected_blocks": 2,  # Should split
                "expected_type": "section"
            },
            {
                "name": "Mixed complex content", 
                "input": "# Report\n\n**Summary:** Important findings\n\n```json\n{\"key\": \"value\"}\n```\n\n- Point 1\n- Point 2",
                "expected_blocks": 4,
                "expected_types": ["header", "section", "section", "section"]
            }
        ]
        
        results = []
        for test_case in test_cases:
            start_time = time.time()
            try:
                blocks = MarkdownToSlackParser.parse_to_blocks(test_case["input"])
                execution_time = (time.time() - start_time) * 1000
                
                # Analyze results
                success = True
                analysis = "SUCCESS"
                
                # Check block count
                if "expected_blocks" in test_case:
                    actual_blocks = len(blocks)
                    expected_blocks = test_case["expected_blocks"]
                    if actual_blocks != expected_blocks:
                        success = False
                        analysis = f"Block count mismatch: expected {expected_blocks}, got {actual_blocks}"
                
                # Check block types
                if success and "expected_types" in test_case:
                    actual_types = [block.get("type") for block in blocks]
                    expected_types = test_case["expected_types"]
                    if actual_types != expected_types:
                        success = False
                        analysis = f"Block types mismatch: expected {expected_types}, got {actual_types}"
                
                # Check content presence
                if success and "expected_content" in test_case:
                    all_text = json.dumps(blocks)
                    if test_case["expected_content"] not in all_text:
                        success = False
                        analysis = f"Expected content '{test_case['expected_content']}' not found"
                
                result = {
                    "test_name": test_case["name"],
                    "input_length": len(test_case["input"]),
                    "output_blocks": len(blocks),
                    "execution_time_ms": round(execution_time, 2),
                    "success": success,
                    "analysis": analysis,
                    "blocks_structure": [{"type": b.get("type"), "has_text": "text" in b} for b in blocks],
                    "timestamp": datetime.now().isoformat()
                }
                
                results.append(result)
                status_icon = "✅" if success else "❌"
                print(f"  {status_icon} {test_case['name']}: {analysis}")
                print(f"      → Input: {len(test_case['input'])} chars → {len(blocks)} blocks ({execution_time:.1f}ms)")
                
            except Exception as e:
                result = {
                    "test_name": test_case["name"],
                    "input_length": len(test_case["input"]),
                    "execution_time_ms": (time.time() - start_time) * 1000,
                    "success": False,
                    "analysis": f"Exception: {str(e)}",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                results.append(result)
                print(f"  ❌ {test_case['name']}: Exception - {e}")
        
        self.test_results.extend(results)
        return results

    async def test_streaming_handler_display(self):
        """Test streaming handler message display updates"""
        print(f"\n🔧 Test: Streaming Handler Display Updates")
        
        channel_id = "C1234567890"
        thread_ts = "1234567890.123456"
        
        # Create streaming handler with mock client
        handler = SlackStreamingHandler(self.mock_client, channel_id, thread_ts, self.user_id)
        
        test_scenarios = [
            {
                "name": "Initial streaming start",
                "action": lambda: handler.start_streaming(),
                "expected_calls": ["chat_postMessage"],
                "expected_content": "Reasoning..."
            },
            {
                "name": "Thinking updates",
                "action": lambda: handler.update_thinking("I need to analyze this request carefully"),
                "expected_calls": ["chat_update"],
                "expected_content": "_I need to analyze this request carefully_"
            },
            {
                "name": "Tool execution start",
                "action": lambda: handler.start_tool("memory.search", {"query": "test"}),
                "expected_calls": ["chat_update"],
                "expected_content": "memory.search"
            },
            {
                "name": "Tool completion with result",
                "action": lambda: handler.complete_tool("Found 5 relevant memories"),
                "expected_calls": ["chat_update"],
                "expected_content": "_Found 5 relevant memories_"
            },
            {
                "name": "Multiple thinking blocks",
                "action": lambda: asyncio.gather(
                    handler.update_thinking("First thought"),
                    handler.update_thinking("Second thought"),
                    handler.update_thinking("Third thought")
                ),
                "expected_calls": ["chat_update"],
                "expected_multiple": True
            }
        ]
        
        results = []
        
        # Set up initial message timestamp for handler
        handler.message_ts = "1234567890.123456"
        
        for scenario in test_scenarios:
            start_time = time.time()
            try:
                # Reset mock call counts
                self.mock_client.reset_mock()
                
                # Execute scenario
                await scenario["action"]()
                execution_time = (time.time() - start_time) * 1000
                
                # Analyze mock calls
                success = True
                analysis = "SUCCESS"
                
                # Check if expected API calls were made
                for expected_call in scenario["expected_calls"]:
                    mock_method = getattr(self.mock_client, expected_call)
                    if not mock_method.called:
                        success = False
                        analysis = f"Expected {expected_call} not called"
                        break
                
                # Check content if applicable
                if success and "expected_content" in scenario:
                    call_args = None
                    if self.mock_client.chat_update.called:
                        call_args = self.mock_client.chat_update.call_args
                    elif self.mock_client.chat_postMessage.called:
                        call_args = self.mock_client.chat_postMessage.call_args
                    
                    if call_args:
                        call_content = json.dumps(call_args.kwargs)
                        if scenario["expected_content"] not in call_content:
                            success = False
                            analysis = f"Expected content '{scenario['expected_content']}' not found in call"
                
                result = {
                    "test_name": scenario["name"],
                    "execution_time_ms": round(execution_time, 2),
                    "success": success,
                    "analysis": analysis,
                    "api_calls_made": {
                        "chat_postMessage": self.mock_client.chat_postMessage.call_count,
                        "chat_update": self.mock_client.chat_update.call_count
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                results.append(result)
                status_icon = "✅" if success else "❌"
                print(f"  {status_icon} {scenario['name']}: {analysis}")
                print(f"      → API calls: post={self.mock_client.chat_postMessage.call_count}, update={self.mock_client.chat_update.call_count} ({execution_time:.1f}ms)")
                
            except Exception as e:
                result = {
                    "test_name": scenario["name"],
                    "execution_time_ms": (time.time() - start_time) * 1000,
                    "success": False,
                    "analysis": f"Exception: {str(e)}",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                results.append(result)
                print(f"  ❌ {scenario['name']}: Exception - {e}")
        
        self.test_results.extend(results)
        return results

    async def test_error_display_handling(self):
        """Test error message display and formatting"""
        print(f"\n🔧 Test: Error Display Handling")
        
        # Test error scenarios
        error_scenarios = [
            {
                "name": "Rate limit error",
                "error_message": "Rate limit exceeded. Please try again in 60 seconds.",
                "expected_format": "rate limit",
                "error_type": "rate_limit"
            },
            {
                "name": "Authentication error",
                "error_message": "Invalid authentication token",
                "expected_format": "authentication",
                "error_type": "auth"
            },
            {
                "name": "Tool execution error",
                "error_message": "Tool 'memory.search' failed: Connection timeout",
                "expected_format": "Tool",
                "error_type": "tool"
            },
            {
                "name": "Parsing error",
                "error_message": "Failed to parse response: Invalid JSON",
                "expected_format": "parse",
                "error_type": "parsing"
            }
        ]
        
        results = []
        
        for scenario in error_scenarios:
            start_time = time.time()
            try:
                # Test markdown parsing of error message
                error_text = f"❌ **Error Processing Request**\n\nSorry, I encountered an error:\n\n`{scenario['error_message']}`"
                blocks = MarkdownToSlackParser.parse_to_blocks(error_text)
                execution_time = (time.time() - start_time) * 1000
                
                # Analyze error formatting
                success = True
                analysis = "SUCCESS"
                
                # Check that blocks were created
                if not blocks:
                    success = False
                    analysis = "No blocks generated for error message"
                else:
                    # Check for error indicators
                    all_content = json.dumps(blocks)
                    if "❌" not in all_content and "Error" not in all_content:
                        success = False
                        analysis = "Error formatting missing error indicators"
                    
                    # Check that original error message is preserved
                    if scenario["expected_format"] not in all_content.lower():
                        success = False
                        analysis = f"Expected error type '{scenario['expected_format']}' not found"
                
                result = {
                    "test_name": scenario["name"],
                    "error_type": scenario["error_type"],
                    "execution_time_ms": round(execution_time, 2),
                    "success": success,
                    "analysis": analysis,
                    "blocks_generated": len(blocks),
                    "has_error_formatting": "❌" in json.dumps(blocks),
                    "timestamp": datetime.now().isoformat()
                }
                
                results.append(result)
                status_icon = "✅" if success else "❌"
                print(f"  {status_icon} {scenario['name']}: {analysis}")
                print(f"      → Generated {len(blocks)} blocks with error formatting ({execution_time:.1f}ms)")
                
            except Exception as e:
                result = {
                    "test_name": scenario["name"],
                    "error_type": scenario["error_type"],
                    "execution_time_ms": (time.time() - start_time) * 1000,
                    "success": False,
                    "analysis": f"Exception: {str(e)}",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                results.append(result)
                print(f"  ❌ {scenario['name']}: Exception - {e}")
        
        self.test_results.extend(results)
        return results

    async def test_modal_display_formatting(self):
        """Test modal display formatting and structure"""
        print(f"\n🔧 Test: Modal Display Formatting")
        
        # Create modal handler with cache service
        cache_service = SlackCacheService()
        modal_handler = SlackModalHandler(cache_service)
        
        # Test execution details scenarios
        execution_scenarios = [
            {
                "name": "Simple execution flow",
                "execution_details": [
                    ("thinking", "I need to search for information"),
                    ("tool", {"name": "memory.search", "operations": ["Searching memory"], "status": "completed"})
                ],
                "expected_blocks": 2,
                "expected_content": ["thinking", "memory.search"]
            },
            {
                "name": "Complex multi-tool execution",
                "execution_details": [
                    ("thinking", "First, I'll search memory"),
                    ("tool", {"name": "memory.search", "operations": ["Found 3 items"], "status": "completed"}),
                    ("thinking", "Now I'll search Slack"),
                    ("tool", {"name": "slack.search", "operations": ["Found 5 messages"], "status": "completed"})
                ],
                "expected_blocks": 4,
                "expected_content": ["memory.search", "slack.search"]
            },
            {
                "name": "Long content requiring pagination",
                "execution_details": [
                    ("thinking", "A" * 1000),  # Long thinking
                    ("tool", {"name": "test.tool", "operations": ["B" * 1000], "status": "completed"})
                ],
                "expected_blocks": 2,
                "expected_pagination": True
            }
        ]
        
        results = []
        
        for scenario in execution_scenarios:
            start_time = time.time()
            try:
                # Build execution pages using modal handler
                pages = modal_handler._build_execution_pages(scenario["execution_details"])
                execution_time = (time.time() - start_time) * 1000
                
                # Analyze modal structure
                success = True
                analysis = "SUCCESS"
                
                # Check page structure
                if not pages:
                    success = False
                    analysis = "No pages generated"
                else:
                    total_blocks = sum(len(page) for page in pages)
                    
                    # Check expected content is present
                    all_content = json.dumps(pages)
                    for expected in scenario.get("expected_content", []):
                        if expected not in all_content:
                            success = False
                            analysis = f"Expected content '{expected}' not found"
                            break
                    
                    # Check pagination if expected
                    if scenario.get("expected_pagination") and len(pages) == 1:
                        success = False
                        analysis = "Expected pagination but only one page generated"
                
                result = {
                    "test_name": scenario["name"],
                    "execution_time_ms": round(execution_time, 2),
                    "success": success,
                    "analysis": analysis,
                    "pages_generated": len(pages),
                    "total_blocks": sum(len(page) for page in pages),
                    "avg_blocks_per_page": sum(len(page) for page in pages) / len(pages) if pages else 0,
                    "timestamp": datetime.now().isoformat()
                }
                
                results.append(result)
                status_icon = "✅" if success else "❌"
                print(f"  {status_icon} {scenario['name']}: {analysis}")
                print(f"      → Generated {len(pages)} pages with {sum(len(page) for page in pages)} total blocks ({execution_time:.1f}ms)")
                
            except Exception as e:
                result = {
                    "test_name": scenario["name"],
                    "execution_time_ms": (time.time() - start_time) * 1000,
                    "success": False,
                    "analysis": f"Exception: {str(e)}",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                results.append(result)
                print(f"  ❌ {scenario['name']}: Exception - {e}")
        
        self.test_results.extend(results)
        return results

    async def test_mention_resolution_display(self):
        """Test mention resolution and display formatting"""
        print(f"\n🔧 Test: Mention Resolution Display")
        
        # Create services for mention resolution
        cache_service = SlackCacheService()
        user_service = SlackUserService(self.mock_client)
        mention_resolver = SlackMentionResolver(cache_service, user_service)
        
        # Test mention scenarios
        mention_scenarios = [
            {
                "name": "User mention",
                "input": "Hey <@U123456789> how are you?",
                "expected_parsing": True,
                "expected_users": ["U123456789"]
            },
            {
                "name": "Channel mention",
                "input": "Check <#C123456789|general> for updates",
                "expected_parsing": True,
                "expected_channels": ["C123456789"]
            },
            {
                "name": "Multiple mentions",
                "input": "CC: <@U111111111> <@U222222222> please review <#C123456789|engineering>",
                "expected_parsing": True,
                "expected_users": ["U111111111", "U222222222"],
                "expected_channels": ["C123456789"]
            },
            {
                "name": "No mentions",
                "input": "Regular message with no mentions",
                "expected_parsing": True,
                "expected_users": [],
                "expected_channels": []
            }
        ]
        
        results = []
        
        for scenario in mention_scenarios:
            start_time = time.time()
            try:
                # Mock user and channel info responses
                self.mock_client.users_info.return_value = {
                    "ok": True,
                    "user": {"id": "U123456789", "name": "testuser", "real_name": "Test User"}
                }
                self.mock_client.conversations_info.return_value = {
                    "ok": True,
                    "channel": {"id": "C123456789", "name": "general"}
                }
                
                # Parse mentions
                parsed_text, mention_context = await mention_resolver.parse_slack_mentions(
                    scenario["input"], self.mock_client
                )
                execution_time = (time.time() - start_time) * 1000
                
                # Analyze parsing results
                success = True
                analysis = "SUCCESS"
                
                # Check basic parsing
                if not scenario["expected_parsing"]:
                    if parsed_text != scenario["input"]:
                        success = False
                        analysis = "Unexpected parsing occurred"
                else:
                    # Check user mentions
                    expected_users = scenario.get("expected_users", [])
                    found_users = mention_context.get("users", [])
                    if len(found_users) != len(expected_users):
                        success = False
                        analysis = f"User mentions mismatch: expected {len(expected_users)}, found {len(found_users)}"
                    
                    # Check channel mentions
                    expected_channels = scenario.get("expected_channels", [])
                    found_channels = mention_context.get("channels", [])
                    if success and len(found_channels) != len(expected_channels):
                        success = False
                        analysis = f"Channel mentions mismatch: expected {len(expected_channels)}, found {len(found_channels)}"
                
                result = {
                    "test_name": scenario["name"],
                    "input_text": scenario["input"],
                    "parsed_text": parsed_text,
                    "execution_time_ms": round(execution_time, 2),
                    "success": success,
                    "analysis": analysis,
                    "users_found": len(mention_context.get("users", [])),
                    "channels_found": len(mention_context.get("channels", [])),
                    "timestamp": datetime.now().isoformat()
                }
                
                results.append(result)
                status_icon = "✅" if success else "❌"
                print(f"  {status_icon} {scenario['name']}: {analysis}")
                print(f"      → Found {len(mention_context.get('users', []))} users, {len(mention_context.get('channels', []))} channels ({execution_time:.1f}ms)")
                
            except Exception as e:
                result = {
                    "test_name": scenario["name"],
                    "input_text": scenario["input"],
                    "execution_time_ms": (time.time() - start_time) * 1000,
                    "success": False,
                    "analysis": f"Exception: {str(e)}",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                results.append(result)
                print(f"  ❌ {scenario['name']}: Exception - {e}")
        
        self.test_results.extend(results)
        return results

    async def analyze_results(self):
        """Analyze overall test results"""
        print("\n" + "=" * 80)
        print("📊 SLACK INTERFACE DISPLAY TEST ANALYSIS")
        print("=" * 80)
        
        # Categorize results by test type
        categories = {}
        for result in self.test_results:
            # Extract category from test name or use a generic key
            if "markdown" in result.get("test_name", "").lower():
                category = "Markdown Conversion"
            elif "streaming" in result.get("test_name", "").lower():
                category = "Streaming Handler"
            elif "error" in result.get("test_name", "").lower():
                category = "Error Display"
            elif "modal" in result.get("test_name", "").lower():
                category = "Modal Formatting"
            elif "mention" in result.get("test_name", "").lower():
                category = "Mention Resolution"
            else:
                category = "Other"
            
            if category not in categories:
                categories[category] = []
            categories[category].append(result)
        
        total_tests = len(self.test_results)
        successful_tests = [r for r in self.test_results if r["success"]]
        failed_tests = [r for r in self.test_results if not r["success"]]
        
        print(f"📈 OVERALL RESULTS:")
        print(f"   Total display tests: {total_tests}")
        print(f"   Successful: {len(successful_tests)} ({len(successful_tests)/total_tests*100:.1f}%)")
        print(f"   Failed: {len(failed_tests)} ({len(failed_tests)/total_tests*100:.1f}%)")
        
        if successful_tests:
            avg_time = sum(r["execution_time_ms"] for r in successful_tests) / len(successful_tests)
            print(f"   Average execution time: {avg_time:.1f}ms")
        
        # Analyze by category
        print(f"\n📋 RESULTS BY CATEGORY:")
        for category, cat_results in categories.items():
            successful = [r for r in cat_results if r["success"]]
            print(f"   {category}: {len(successful)}/{len(cat_results)} successful ({len(successful)/len(cat_results)*100:.1f}%)")
        
        # Performance analysis
        if self.test_results:
            times = [r["execution_time_ms"] for r in self.test_results]
            print(f"\n⚡ PERFORMANCE ANALYSIS:")
            print(f"   Fastest: {min(times):.1f}ms")
            print(f"   Slowest: {max(times):.1f}ms")
            print(f"   Average: {sum(times)/len(times):.1f}ms")
        
        # Display quality analysis
        print(f"\n🎨 DISPLAY QUALITY INDICATORS:")
        
        # Markdown conversion quality
        markdown_tests = [r for r in self.test_results if "markdown" in r.get("test_name", "").lower()]
        if markdown_tests:
            successful_markdown = [r for r in markdown_tests if r["success"]]
            print(f"   Markdown Conversion: {len(successful_markdown)}/{len(markdown_tests)} tests passed")
            
        # Block generation analysis
        block_tests = [r for r in self.test_results if "blocks" in str(r)]
        if block_tests:
            avg_blocks = sum(r.get("output_blocks", r.get("blocks_generated", 0)) for r in block_tests) / len(block_tests)
            print(f"   Average blocks per conversion: {avg_blocks:.1f}")
        
        # Error handling quality
        error_tests = [r for r in self.test_results if "error" in r.get("test_name", "").lower()]
        if error_tests:
            successful_errors = [r for r in error_tests if r["success"]]
            print(f"   Error Display Handling: {len(successful_errors)}/{len(error_tests)} scenarios handled properly")
        
        await self.save_test_results()

    async def save_test_results(self):
        """Save detailed test results to JSON file"""
        filename = f"tests/slack_interface_display_results_{self.user_id}.json"
        
        test_report = {
            "test_metadata": {
                "test_type": "slack_interface_display_test",
                "user_id": self.user_id,
                "total_tests": len(self.test_results),
                "successful_tests": len([r for r in self.test_results if r["success"]]),
                "test_duration_ms": sum(r["execution_time_ms"] for r in self.test_results),
                "timestamp": datetime.now().isoformat()
            },
            "categories": {
                "markdown_conversion": len([r for r in self.test_results if "markdown" in r.get("test_name", "").lower()]),
                "streaming_display": len([r for r in self.test_results if "streaming" in r.get("test_name", "").lower()]),
                "error_handling": len([r for r in self.test_results if "error" in r.get("test_name", "").lower()]),
                "modal_formatting": len([r for r in self.test_results if "modal" in r.get("test_name", "").lower()]),
                "mention_resolution": len([r for r in self.test_results if "mention" in r.get("test_name", "").lower()])
            },
            "summary": {
                "success_rate": len([r for r in self.test_results if r["success"]]) / len(self.test_results) if self.test_results else 0,
                "avg_execution_time_ms": sum(r["execution_time_ms"] for r in self.test_results) / len(self.test_results) if self.test_results else 0,
                "display_quality_score": self._calculate_display_quality_score()
            },
            "detailed_results": self.test_results
        }
        
        with open(filename, 'w') as f:
            json.dump(test_report, f, indent=2, default=str)
        
        print(f"\n💾 Detailed test results saved to: {filename}")

    def _calculate_display_quality_score(self):
        """Calculate overall display quality score"""
        if not self.test_results:
            return 0.0
        
        # Weight different types of tests
        weights = {
            "markdown": 0.3,  # Critical for message display
            "streaming": 0.25,  # Important for user experience
            "error": 0.2,     # Important for error handling
            "modal": 0.15,    # Important for detailed views
            "mention": 0.1    # Nice to have
        }
        
        category_scores = {}
        for weight_key, weight in weights.items():
            category_tests = [r for r in self.test_results if weight_key in r.get("test_name", "").lower()]
            if category_tests:
                success_rate = len([r for r in category_tests if r["success"]]) / len(category_tests)
                category_scores[weight_key] = success_rate * weight
            else:
                category_scores[weight_key] = 0
        
        return sum(category_scores.values())


async def main():
    """Run the Slack interface display test suite"""
    try:
        test = SlackInterfaceDisplayTest()
        
        print("🎯 SLACK INTERFACE DISPLAY TEST SUITE")
        print("=" * 60)
        print(f"Testing visual display components, formatting, and user experience")
        print(f"Test User ID: {test.user_id}")
        print(f"Start Time: {datetime.now().isoformat()}")
        
        # Setup
        if not await test.setup():
            print("❌ Setup failed, aborting tests")
            return None
        
        # Run all test categories
        print(f"\n🧪 Running Display Tests...")
        
        await test.test_markdown_to_slack_conversion()
        await test.test_streaming_handler_display()
        await test.test_error_display_handling()
        await test.test_modal_display_formatting()
        await test.test_mention_resolution_display()
        
        # Analyze results
        await test.analyze_results()
        
        print("\n🎉 Slack Interface Display Tests COMPLETED!")
        
        # Final summary
        total = len(test.test_results)
        successful = len([r for r in test.test_results if r["success"]])
        quality_score = test._calculate_display_quality_score()
        
        print(f"📄 Final Summary:")
        print(f"   - Total display tests: {total}")
        print(f"   - Successful: {successful}/{total} ({successful/total*100:.1f}%)")
        print(f"   - Display Quality Score: {quality_score:.2f}/1.00")
        print(f"   - Test User ID: {test.user_id}")
        
        return test.test_results
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the Slack interface display test suite
    asyncio.run(main())
