import asyncio
import logging
from datetime import datetime, timezone

from agents.client_agent import ClientAgent
from tools.memory_mcp import close_mcp_client


async def main() -> None:
    logging.basicConfig(level=logging.INFO)

    agent = ClientAgent()

    context = {
        "platform": "slack",
        "user_id": "U_LOCAL_TEST",
        "channel_id": "C_LOCAL_TEST",
        "thread_ts": "123.456",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "user_timezone": "UTC",
    }

    message = "Hello! Please retrieve recent conversation history first, then answer: what is 2+3? and store this exchange to memory."

    print("Running ClientAgent.process_request with Slack-like context...\n")
    result = await agent.process_request(message, context=context, streaming_callback=None)
    print("\nResult summary:\n", result)

    # Ensure MCP is cleanly closed to avoid noisy shutdown logs
    await close_mcp_client()


if __name__ == "__main__":
    asyncio.run(main())


