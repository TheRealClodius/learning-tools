import asyncio
import logging
import sys
import argparse
import time
from typing import Dict, Any, Optional
from datetime import datetime
import signal
from dotenv import load_dotenv

# Load environment variables from .env.local (override=True to pick up changes)
load_dotenv('.env.local', override=True)

# Import agent and runtime components
from agents.client_agent import ClientAgent
from runtime.tool_executor import ToolExecutor
from runtime.rate_limit_handler import RateLimitError

# Configure logging for CLI
logging.basicConfig(
    level=logging.WARNING,  # Less verbose for CLI
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class CLIInterface:
    """
    Command Line Interface for agent interactions
    
    Provides an interactive command-line interface for communicating
    with the agent, supporting both interactive and single-command modes.
    """
    
    def __init__(self, verbose: bool = False):
        self.agent = ClientAgent()
        self.tool_executor = ToolExecutor()
        self.verbose = verbose
        self.running = True
        self.thinking_active = False  # Track streaming thinking state
        
        # Generate unique user ID for this CLI session
        import uuid
        import time
        self.user_id = f"cli_user_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        print(f"🆔 Session User ID: {self.user_id}")
        
        # Setup logging level based on verbose flag
        if verbose:
            logging.getLogger().setLevel(logging.INFO)
        else:
            # In non-verbose mode, suppress most logs to keep output clean
            # Only show ERROR level logs to avoid cluttering the UI
            logging.getLogger().setLevel(logging.ERROR)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print("\n\n🔄 Shutting down agent CLI...")
        self.running = False
    
    async def interactive_mode(self):
        """Run interactive CLI mode"""
        self._print_welcome()
        
        try:
            while self.running:
                try:
                    # Get user input
                    user_input = await self._get_user_input()
                    
                    if not user_input:
                        continue
                    
                    # Handle special commands
                    if user_input.lower() in ['exit', 'quit', 'q']:
                        break
                    elif user_input.lower() in ['help', 'h']:
                        self._print_help()
                        continue
                    elif user_input.lower() in ['status']:
                        await self._print_status()
                        continue
                    elif user_input.lower() in ['tools']:
                        await self._print_available_tools()
                        continue
                    elif user_input.lower().startswith('clear'):
                        self._clear_screen()
                        continue
                    
                    # Process agent request
                    await self._process_user_message(user_input)
                    
                except KeyboardInterrupt:
                    print("\n\n👋 Goodbye!")
                    break
                except Exception as e:
                    print(f"❌ Error: {str(e)}")
                    if self.verbose:
                        logger.exception("CLI error details:")
        finally:
            # Cleanup on exit
            print("\n🧹 Cleaning up resources...")
            await self.agent.cleanup()
            print("✅ Cleanup complete")
    
    async def single_command_mode(self, command: str):
        """Process a single command and exit"""
        try:
            if self.verbose:
                print(f"🤖 Processing: {command}")
            
            await self._process_user_message(command)
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            if self.verbose:
                logger.exception("Command execution error:")
            sys.exit(1)
        finally:
            # Cleanup on exit
            await self.agent.cleanup()
    
    async def _get_user_input(self) -> str:
        """Get user input asynchronously"""
        try:
            # Ensure we're ready for input by flushing any pending output
            import sys
            sys.stdout.flush()
            sys.stderr.flush()
            
            # Add small delay to ensure background tasks have completed
            await asyncio.sleep(0.1)
            
            # Run input in thread to avoid blocking
            loop = asyncio.get_event_loop()
            user_input = await loop.run_in_executor(
                None, 
                lambda: input("💬 You: ").strip()
            )
            return user_input
        except EOFError:
            return "exit"
        except Exception as e:
            if self.verbose:
                print(f"\n🔍 DEBUG - Input error: {e}")
            return "exit"
    
    async def _process_user_message(self, message: str):
        """Process user message through agent"""
        try:
            # Initialize streaming state
            self.thinking_active = True
            start_time = time.time()
            
            # Show prettier status in non-verbose mode
            if not self.verbose:
                print("🎯 Processing your request...")
            else:
                print("Reasoning...")
            
            # Get timestamp for context
            timestamp = datetime.now().isoformat()
            
            # Process through agent with streaming callback
            response = await self.agent.process_request(
                message,
                context={
                    "platform": "cli",
                    "user_id": self.user_id,  # Unique user ID for this session
                    "timestamp": timestamp,
                    "verbose": self.verbose,
                    "user_timezone": "UTC",
                    "user_name": "CLI User",
                    # Disable insights updates in CLI to avoid background hangs by default
                    "disable_insights": True
                },
                streaming_callback=self._streaming_callback
            )
            
            # End thinking phase
            self.thinking_active = False
            total_time = int((time.time() - start_time) * 1000)
            
            # Show routing information
            agent_used = response.get("agent", "unknown")
            if not self.verbose:
                # Prettier, simpler output for non-verbose mode
                if agent_used.startswith("gemini"):
                    print(f"\n🟢 Gemini • {total_time/1000:.1f}s")
                elif agent_used.startswith("claude"):
                    print(f"\n🔵 Claude • {total_time/1000:.1f}s")
                else:
                    print(f"\n✨ Complete • {total_time/1000:.1f}s")
            else:
                # Detailed output for verbose mode
                if agent_used.startswith("gemini"):
                    print(f"\n🟢 Processed by Gemini (Simple) in {total_time}ms")
                elif agent_used.startswith("claude"):
                    print(f"\n🔵 Processed by Claude (Complex) in {total_time}ms")
                else:
                    print(f"\n✨ Ready to respond! (Total processing: {total_time}ms)")
            
            # Display response
            self._display_response(response)
            
            # DEBUG: Add a small delay to ensure all background operations complete
            if self.verbose:
                print("\n🔍 DEBUG - Response display complete, preparing for next input...")
            await asyncio.sleep(0.1)  # Give background tasks time to complete
            
        except RateLimitError as e:
            self.thinking_active = False
            # Rate limit errors already have user-friendly messages
            print(f"\n{str(e)}")
            
        except Exception as e:
            self.thinking_active = False
            # Check if it's a rate limit error that wasn't caught
            if 'rate_limit_error' in str(e).lower() or '429' in str(e):
                print("\n⏳ The service is experiencing high demand. Please try again in a moment.")
            else:
                print(f"\n❌ Agent Error: {str(e)}")
            if self.verbose:
                logger.exception("Agent processing error:")
    
    async def _streaming_callback(self, content: str, content_type: str):
        """Handle streaming content from agent - only actual events sent"""
        if content_type == "thinking" and self.thinking_active:
            if self.verbose:
                # In verbose mode, show detailed thinking
                print(f"\n💭 {content}")
                await asyncio.sleep(0.2)
            # In non-verbose mode, just update a simple thinking indicator
            elif not hasattr(self, '_thinking_dots'):
                self._thinking_dots = 0
                print("🤔 Thinking", end="", flush=True)
            else:
                self._thinking_dots = (self._thinking_dots + 1) % 4
                print("." * (self._thinking_dots + 1) + "\b" * 4, end="", flush=True)
                await asyncio.sleep(0.3)
        elif content_type == "tool_start":
            # Only show tool_start in verbose mode or when execution summaries are disabled
            # In normal mode, execution summaries will replace this notification
            if self.verbose:
                if isinstance(content, dict):
                    tool_name = content.get("name", "unknown")
                    print(f"\n🚀 Using {tool_name}...")
                else:
                    print(f"\n🚀 Using {content}...")
            # In non-verbose mode, suppress tool_start - execution summaries will show instead
        elif content_type == "tool_summary_chunk":
            # MANDATORY: Real-time Gemini summary chunks from ExecutionSummarizer
            # This replaces operation + tool_result - it's the live AI-generated summary
            print(content, end="", flush=True)
    
    def _display_response(self, response: Dict[str, Any]):
        """Display agent response in formatted way with routing and execution info"""
        print("\n🤖 Agent Response:")
        
        # Main response content
        agent_response = response.get("response", "")
        if agent_response:
            # Format multi-line responses nicely
            lines = agent_response.split('\n')
            for line in lines:
                if line.strip():
                    print(f"   {line}")
                else:
                    print()
        
        # Get agent and model info (needed for both verbose and non-verbose)
        agent_used = response.get("agent", "unknown")
        model_used = response.get("model", "unknown")
        
        # Show execution summary (only in verbose mode)
        if self.verbose:
            
            print(f"\n📊 Execution Summary:")
            print(f"   • Agent: {agent_used}")
            print(f"   • Model: {model_used}")
            
            # Show processing times
            routing_time = response.get("routing_time_ms")
            processing_time = response.get("processing_time_ms")
            total_time = response.get("total_time_ms")
            
            if routing_time is not None:
                print(f"   • Routing time: {routing_time}ms")
            if processing_time is not None:
                print(f"   • Processing time: {processing_time}ms")
            if total_time is not None:
                print(f"   • Total time: {total_time}ms")
        
        # Processing message  
        message = response.get("message", "")
        if message and message != "Request processed successfully":
            print(f"   • Status: {message}")
        
        # Debug: Show what's actually in the response
        if self.verbose:
            print(f"\n🔍 DEBUG - Response keys: {list(response.keys())}")
            if "tool_calls" in response:
                print(f"🔍 DEBUG - Tool calls found: {len(response['tool_calls'])}")
            else:
                print("🔍 DEBUG - No 'tool_calls' key in response")
        
        # Tool calls information - always show with detailed timing
        tool_calls = response.get("tool_calls", [])
        if tool_calls:
            print("\n🔧 Tools Executed:")
            for i, tool_call in enumerate(tool_calls, 1):
                tool_name = tool_call.get("tool", "unknown")
                execution_time = tool_call.get("execution_time_ms", 0)
                result_preview = tool_call.get("result_preview", "")
                success = tool_call.get("success", True)
                
                # Show tool with timing and status
                status_icon = "✅" if success else "❌"
                print(f"   {i}. {status_icon} {tool_name}: {execution_time}ms")
                
                # Show brief result preview
                if result_preview:
                    preview = result_preview[:150] + "..." if len(result_preview) > 150 else result_preview
                    print(f"      → {preview}")
                elif not success:
                    error_msg = tool_call.get("error", "Unknown error")
                    print(f"      → Error: {error_msg}")
        # Only show tool information if tools were actually used
        
        # Check for memory usage indicators
        if any(keyword in agent_response.lower() for keyword in ["remember", "recall", "previous", "context", "history"]):
            print("\n🧠 Memory: Conversation context utilized")
        
        # Context information
        context = response.get("context", {})
        if context and self.verbose:
            print("\n📝 Context:")
            for key, value in context.items():
                if isinstance(value, str) and len(value) < 100:
                    print(f"   • {key}: {value}")
        
        print()  # Empty line for spacing
    
    async def _print_status(self):
        """Print agent and tool status"""
        print("\n📊 Agent Status:")
        print(f"   • Agent Type: {self.agent.__class__.__name__}")
        print(f"   • Available Tools: {len(self.tool_executor.available_tools)}")
        print(f"   • Loaded Services: {', '.join(self.tool_executor.loaded_services)}")
        
        # List loaded tools
        if self.verbose and self.tool_executor.available_tools:
            print("\n🔧 Loaded Tools:")
            for tool_name in sorted(self.tool_executor.available_tools.keys()):
                print(f"   • {tool_name}")
        
        print()
    
    async def _print_available_tools(self):
        """Print available tools from registry"""
        try:
            print("\n🔍 Discovering available tools...")
            
            # Query registry for all tools
            tools = await self.tool_executor.discover_tools("", None)
            
            if tools:
                print(f"\n🔧 Available Tools ({len(tools)}):")
                
                # Group by category
                categories = {}
                for tool in tools:
                    category = tool.get("category", "other")
                    if category not in categories:
                        categories[category] = []
                    categories[category].append(tool)
                
                # Display by category
                for category, category_tools in sorted(categories.items()):
                    print(f"\n   📂 {category.title()}:")
                    for tool in category_tools:
                        name = tool.get("name", "unknown")
                        description = tool.get("description", "No description")
                        if isinstance(description, list):
                            description = description[0] if description else "No description"
                        print(f"      • {name}: {description[:60]}...")
            else:
                print("   No tools found in registry")
            
        except Exception as e:
            print(f"❌ Error discovering tools: {str(e)}")
        
        print()
    
    def _print_welcome(self):
        """Print welcome message"""
        print("=" * 60)
        print("🤖 AI Agent CLI Interface")
        print("=" * 60)
        print("Welcome! I'm your intelligent AI assistant with routing capabilities.")
        print("I can route to Gemini (fast) or Claude (complex) based on your needs.")
        print("\nAvailable commands:")
        print("  • help/h      - Show this help")
        print("  • status      - Show agent status") 
        print("  • tools       - List available tools")
        print("  • clear       - Clear screen")
        print("  • exit/quit/q - Exit")
        print("\nFeatures:")
        print("  🎯 Smart routing (Gemini for simple, Claude for complex tasks)")
        print("  🔧 Dynamic tool execution (weather, search, memory)")
        print("  ⚡ Real-time execution summaries powered by AI")
        print("  🧠 Conversation memory across interactions")
        print("\nTip: Use --verbose flag for detailed streaming information")
        print("=" * 60)
        print()
    
    def _print_help(self):
        """Print help information"""
        print("\n📚 Help - AI Agent CLI")
        print("=" * 40)
        print("Commands:")
        print("  help, h           - Show this help message")
        print("  status            - Show agent and tool status")
        print("  tools             - List all available tools")
        print("  clear             - Clear the screen")
        print("  exit, quit, q     - Exit the CLI")
        print("\nUsage Examples:")
        print("  💬 What's the weather in London?")
        print("  💬 Search for information about Python")
        print("  💬 Help me with a coding problem")
        print("\nTips:")
        print("  • Be specific in your requests")
        print("  • Use --verbose for detailed logs")
        print("  • The agent can use multiple tools to help you")
        print("=" * 40)
        print()
    
    def _clear_screen(self):
        """Clear the terminal screen"""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        self._print_welcome()

def create_argument_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="AI Agent CLI Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_interface.py                           # Interactive mode
  python cli_interface.py "What's the weather?"     # Single command
  python cli_interface.py --verbose "Search for AI" # Verbose single command
        """
    )
    
    parser.add_argument(
        "command",
        nargs="?",
        help="Single command to execute (if not provided, enters interactive mode)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output with detailed logs"
    )
    
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Force interactive mode even if command is provided"
    )
    
    return parser

async def main():
    """Main CLI entry point"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Create CLI interface
    cli = CLIInterface(verbose=args.verbose)
    
    try:
        if args.command and not args.interactive:
            # Single command mode
            await cli.single_command_mode(args.command)
        else:
            # Interactive mode
            await cli.interactive_mode()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        if args.verbose:
            logger.exception("Fatal error details:")
        sys.exit(1)

if __name__ == "__main__":
    # Ensure proper handling of asyncio on different platforms
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main()) 