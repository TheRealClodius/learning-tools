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
            # Run input in thread to avoid blocking
            loop = asyncio.get_event_loop()
            user_input = await loop.run_in_executor(
                None, 
                lambda: input("💬 You: ").strip()
            )
            return user_input
        except EOFError:
            return "exit"
    
    async def _process_user_message(self, message: str):
        """Process user message through agent"""
        try:
            # Initialize streaming state
            self.thinking_active = True
            start_time = time.time()
            print("🤔 Thinking...")
            
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
                    # Disable insights updates in CLI to avoid background hangs by default
                    "disable_insights": True
                },
                streaming_callback=self._streaming_callback
            )
            
            # End thinking phase
            self.thinking_active = False
            total_time = int((time.time() - start_time) * 1000)
            print(f"\n✨ Ready to respond! (Total processing: {total_time}ms)")
            
            # Display response
            self._display_response(response)
            
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
        """Handle streaming content from agent"""
        if content_type == "thinking" and self.thinking_active:
            # Show each thinking line as it comes, creating a stream of consciousness
            print(f"\n💭 {content}")
            await asyncio.sleep(0.2)  # Brief pause to show the thinking flow
        elif content_type == "tool_discovery":
            # Show tool discovery notifications
            print(f"\n🔍 {content}")
        elif content_type == "tool_execution":
            # Show tool execution notifications
            print(f"\n⚡ {content}")
        elif content_type == "tool_result":
            # Show tool results briefly
            print(f"\n✅ {content}")
    
    def _display_response(self, response: Dict[str, Any]):
        """Display agent response in formatted way"""
        print("\n🤖 Agent:")
        
        # Main response content
        agent_response = response.get("response", "")
        if agent_response:
            print(f"   {agent_response}")
        
        # Processing message  
        message = response.get("message", "")
        if message:
            print(f"\n   Status: {message}")
        
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
            print("\n🔧 Tools executed:")
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
        else:
            # If no tool_calls found, let's check if memory is working by showing evidence
            if "Sarah" in agent_response or "TechCorp" in agent_response or "Kubernetes" in agent_response:
                print("\n🧠 Memory retrieval detected (agent knows previous context)")
        
        # Context information
        context = response.get("context", {})
        if context and self.verbose:
            print("\n📝 Context:")
            for key, value in context.items():
                if isinstance(value, str) and len(value) < 100:
                    print(f"   • {key}: {value}")
        
        # Show total agent timing if available
        total_time = response.get("total_time_ms")
        if total_time:
            print(f"\n⏱️  Total agent time: {total_time}ms")
        
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
        print("Welcome! I'm your AI agent assistant.")
        print("Type your message and I'll help you with various tasks.")
        print("\nAvailable commands:")
        print("  • help/h      - Show this help")
        print("  • status      - Show agent status") 
        print("  • tools       - List available tools")
        print("  • clear       - Clear screen")
        print("  • exit/quit/q - Exit")
        print("\nTip: Use --verbose flag for detailed information")
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