# |---------------------------------------------------------|
# |                                                         |
# |                 Give Feedback / Get Help                |
# | https://github.com/getbindu/Bindu/issues/new/choose    |
# |                                                         |
# |---------------------------------------------------------|
#
#  Thank you users! We ❤️ you! - 🌻

"""investor-agent - An Bindu Agent."""

import argparse
import asyncio
import json
import os
from pathlib import Path
from textwrap import dedent
from typing import Any

from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.tools import Toolkit
from agno.tools.mem0 import Mem0Tools
from bindu.penguin.bindufy import bindufy
from dotenv import load_dotenv

from investor_agent.tools import (
    calculate_technical_indicator,
    get_cnn_fear_greed_index,
    get_crypto_fear_greed_index,
    get_earnings_history,
    get_financial_statements,
    get_google_trends,
    get_insider_trades,
    get_institutional_holders,
    get_market_movers,
    get_nasdaq_earnings_calendar,
    get_options,
    get_price_history,
    get_ticker_data,
)

# Load environment variables from .env file
load_dotenv()

# Global tools instances
investment_tools: Toolkit | None = None
agent: Agent | None = None
model_name: str | None = None
openrouter_api_key: str | None = None
mem0_api_key: str | None = None
_initialized = False
_init_lock = asyncio.Lock()


class InvestmentTools(Toolkit):
    """Custom toolkit for investment analysis functions."""

    def __init__(self):
        super().__init__(name="investment_tools")
        self.register(get_market_movers)
        self.register(get_cnn_fear_greed_index)
        self.register(get_crypto_fear_greed_index)
        self.register(get_google_trends)
        self.register(get_ticker_data)
        self.register(get_options)
        self.register(get_price_history)
        self.register(get_financial_statements)
        self.register(get_institutional_holders)
        self.register(get_earnings_history)
        self.register(get_insider_trades)
        self.register(get_nasdaq_earnings_calendar)
        self.register(calculate_technical_indicator)


def initialize_investment_tools() -> None:
    """Initialize all investment analysis tools as a Toolkit instance."""
    global investment_tools

    investment_tools = InvestmentTools()
    print("✅ Investment analysis tools initialized")


def load_config() -> dict:
    """Load agent configuration from project root."""
    # Get path to agent_config.json in project root
    config_path = Path(__file__).parent / "agent_config.json"

    with open(config_path) as f:
        return json.load(f)


# Create the agent instance
async def initialize_agent() -> None:
    """Initialize the agent once."""
    global agent, model_name, investment_tools

    if not model_name:
        msg = "model_name must be set before initializing agent"
        raise ValueError(msg)

    # Initialize investment tools if not already done
    if not investment_tools:
        initialize_investment_tools()

    agent = Agent(
        name="Investment Analysis Agent",
        model=OpenRouter(
            id=model_name,
            api_key=openrouter_api_key,
            cache_response=True,
            supports_native_structured_outputs=True,
        ),
        tools=[
            tool
            for tool in [
                investment_tools,  # Investment analysis toolkit
                Mem0Tools(api_key=mem0_api_key) if mem0_api_key else None,
            ]
            if tool is not None
        ],
        description=dedent("""\
            You are an elite financial analyst and investment researcher with decades of experience
            at top investment firms and hedge funds. Your expertise encompasses: 💼

            - Comprehensive financial analysis and valuation
            - Market sentiment analysis and trend identification
            - Risk assessment and portfolio optimization
            - Earnings analysis and insider trading insights
            - Technical analysis and options strategies
            - Institutional ownership analysis
            - Real-time market data interpretation
            - Investment recommendation generation\
        """),
        instructions=dedent("""\
            1. Data Gathering Phase 📊
               - Use investor-agent-org tools to fetch comprehensive financial data
               - Get market movers, ticker data, and sentiment indicators
               - Analyze financial statements and ownership patterns

            2. Analysis Phase 🔍
               - Perform fundamental and technical analysis
               - Evaluate market sentiment and macro trends
               - Assess risks and identify catalysts

            3. Investment Thesis Phase 💡
               - Form clear investment recommendations
               - Provide valuation assessments and price targets
               - Outline key risks and potential scenarios

            4. Reporting Phase 📋
               - Structure analysis in professional investment format
               - Include actionable insights and specific recommendations
               - Provide confidence levels and time horizons

            Always:
            - Use real-time data from integrated investment analysis tools
            - Cite specific metrics and data points
            - Present balanced analysis with risks and opportunities
            - Provide clear buy/sell/hold recommendations with rationale\
        """),
        expected_output=dedent("""\
            # Investment Analysis: {Company/Ticker} 📈

            ## Executive Summary
            {Key investment thesis and recommendation}
            {Target price and confidence level}
            {Investment time horizon}

            ## Company Overview
            {Business description and market position}
            {Key financial metrics and valuation}
            {Competitive advantages and moat}

            ## Financial Analysis
            {Revenue and earnings trends}
            {Profitability and margin analysis}
            {Balance sheet strength and cash flow}
            {Valuation metrics and comparables}

            ## Market Analysis
            {Market sentiment and trends}
            {Institutional ownership and insider activity}
            {Technical analysis and price action}
            {Options market activity}

            ## Investment Thesis
            {Bull case and catalysts}
            {Bear case and risk factors}
            {Valuation methodology and assumptions}

            ## Recommendation
            {Clear buy/sell/hold recommendation}
            {Price target and time horizon}
            {Position sizing suggestions}
            {Key monitoring points}

            ---
            Analysis by AI Investment Analyst
            Data sourced from integrated investment analysis tools
            Generated: {current_date}
            Market data as of: {current_time}\
        """),
        add_datetime_to_context=True,
        markdown=True,
    )
    print("✅ Agent initialized")


async def cleanup_tools() -> None:
    """Clean up any resources."""
    global investment_tools

    if investment_tools:
        print("🔌 Investment analysis tools cleaned up")


async def run_agent(messages: list[dict[str, str]]) -> Any:
    """Run the agent with the given messages.

    Args:
        messages: List of message dicts with 'role' and 'content' keys

    Returns:
        Agent response
    """
    global agent

    # Run the agent and get response
    if agent is None:
        raise ValueError("Agent not initialized")
    response = await agent.arun(messages)
    return response


async def handler(messages: list[dict[str, str]]) -> Any:
    """Handle incoming agent messages.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
                  e.g., [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

    Returns:
        Agent response (ManifestWorker will handle extraction)
    """

    # Run agent with messages
    global _initialized

    # Lazy initialization on first call (in bindufy's event loop)
    async with _init_lock:
        if not _initialized:
            print("🔧 Initializing investment analysis agent...")
            await initialize_agent()
            _initialized = True

    # Run the async agent
    result = await run_agent(messages)
    return result


async def initialize_all(env: dict[str, str] | None = None):
    """Initialize agent and tools.

    Args:
        env: Environment variables dict (not used for integrated tools)
    """
    await initialize_agent()


def main():
    """Run the Investment Analysis Agent."""
    global model_name, openrouter_api_key, mem0_api_key

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Bindu Agent with MCP Tools")
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL_NAME", "anthropic/claude-3.5-sonnet"),
        help="Model ID to use (default: anthropic/claude-3.5-sonnet, env: MODEL_NAME), if you want you can use any free model: https://openrouter.ai/models?q=free",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENROUTER_API_KEY"),
        help="OpenRouter API key (env: OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--mem0-api-key",
        type=str,
        default=os.getenv("MEM0_API_KEY"),
        help="Mem0 API key (env: MEM0_API_KEY)",
    )
    args = parser.parse_args()

    # Set global model name and API keys
    model_name = args.model
    openrouter_api_key = args.api_key
    mem0_api_key = args.mem0_api_key

    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY required")
    if not mem0_api_key:
        raise ValueError("MEM0_API_KEY required. Get your API key from: https://app.mem0.ai/dashboard/api-keys")

    print(f"🤖 Investment Analysis Agent using model: {model_name}")
    print("💼 Comprehensive financial analysis with integrated investment tools")
    print("🧠 Mem0 memory enabled")

    # Load configuration
    config = load_config()

    try:
        # Bindufy and start the agent server
        # Note: Agent will be initialized lazily on first request
        print("🚀 Starting Investment Analysis Agent server...")
        bindufy(config, handler)
    finally:
        # Cleanup on exit
        print("\n🧹 Cleaning up...")
        asyncio.run(cleanup_tools())


# Bindufy and start the agent server
if __name__ == "__main__":
    main()
